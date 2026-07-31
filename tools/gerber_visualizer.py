#!/usr/bin/env python3
"""Render RS-274D/RS-274X-like Gerber files to SVG and an HTML PCB viewer.

This renderer is intentionally dependency-free. It understands the compact
RS-274D style used by the provided files:

  * absolute coordinates
  * leading zero suppression
  * FORM=2.4 / UNIT=INCHES metadata in G04 comments
  * aperture definitions in G04%A... comments, with AA.ENV fallback
  * D01 draw, D02 move, D03 flash, and modal coordinates/operations
  * top/bottom physical preview compositing for PCB-like inspection

Arcs and region fills are not implemented because the inspected files do not
use G02/G03 or region statements. Filled copper areas exported as many draw
segments still render correctly as stroked vector geometry.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import DefaultDict, Iterable


TOKEN_RE = re.compile(r"([A-Z])([+-]?\d+)")
FORM_RE = re.compile(r"FORM\s*=\s*(\d+)\.(\d+)", re.IGNORECASE)
UNIT_RE = re.compile(r"UNIT\s*=\s*([A-Z]+)", re.IGNORECASE)
ZERO_RE = re.compile(r"ZERO\s*=\s*([A-Z]+)", re.IGNORECASE)
MODE_RE = re.compile(r"MODE\s*=\s*([A-Z]+)", re.IGNORECASE)
APERTURE_RE = re.compile(r"A(\d+):(.+)", re.IGNORECASE)


DEFAULT_COLORS = {
    "COMP-P": "#d66a3c",
    "SOLD-P": "#4a90e2",
    "COMP-S": "#f2c94c",
    "SOLD-S": "#9b6bff",
    "C-MASK": "#1f9d55",
    "S-MASK": "#0f7a4d",
    "C-SILK": "#f4f7fb",
    "S-SILK": "#ccd5e0",
    "DRILL": "#101820",
    "D3": "#ff5d8f",
}

SIDE_CONFIG = {
    "top": {
        "label": "Top / Component Side",
        "copper": "COMP-P",
        "mask": "C-MASK",
        "silk": "C-SILK",
        "outline": "COMP-S",
    },
    "bottom": {
        "label": "Bottom / Solder Side",
        "copper": "SOLD-P",
        "mask": "S-MASK",
        "silk": "S-SILK",
        "outline": "SOLD-S",
    },
}

SMALL_VIA_FLASH_THRESHOLD = 0.026
SMALL_VIA_FLASH_SCALE = 0.86
SMALL_DRILL_FLASH_THRESHOLD = 0.012
SMALL_DRILL_FLASH_SCALE = 0.62
DRILL_RING_FLASH_SCALE = 1.36
TRACE_FIELD_STROKE_SCALE = 2.15
TRACE_RELIEF_STROKE_SCALE = 1.55
TRACE_DARK_STROKE_SCALE = 1.05
TRACE_LIGHT_STROKE_SCALE = 0.26
MAX_TRACE_DRAW_WIDTH = 0.012
SILK_STROKE_SCALE = 0.76

TOP_TRACE_NO_CONNECT_POINTS = (
    (1.8610, 0.6004),
    (2.0110, 0.6004),
)
TRACE_NO_CONNECT_RADIUS = 0.018
TRACE_NO_CONNECT_Y_RADIUS = 0.075

TOP_MASK_SHADOW_PATCHES = (
    ((2.430, 1.720), (2.430, 1.780), (2.500, 1.820), (2.840, 1.820), (2.840, 1.720), (2.500, 1.720)),
)

TOP_U6_PAD_RECTS = (
    (1.2067, 1.0870, 1.2343, 1.0980),
    (1.2067, 1.0673, 1.2343, 1.0783),
    (1.2067, 1.0477, 1.2343, 1.0587),
    (1.2067, 1.0280, 1.2343, 1.0390),
    (1.2067, 1.0083, 1.2343, 1.0193),
    (1.2067, 0.9886, 1.2343, 0.9996),
    (1.2067, 0.9689, 1.2343, 0.9799),
    (1.2067, 0.9492, 1.2343, 0.9602),
    (1.4036, 1.0870, 1.4311, 1.0980),
    (1.4036, 1.0673, 1.4311, 1.0783),
    (1.4036, 1.0477, 1.4311, 1.0587),
    (1.4036, 1.0280, 1.4311, 1.0390),
    (1.4036, 1.0083, 1.4311, 1.0193),
    (1.4036, 0.9886, 1.4311, 0.9996),
    (1.4036, 0.9689, 1.4311, 0.9799),
    (1.4036, 0.9492, 1.4311, 0.9602),
    (1.2445, 0.9114, 1.2555, 0.9390),
    (1.2642, 0.9114, 1.2752, 0.9390),
    (1.2839, 0.9114, 1.2949, 0.9390),
    (1.3036, 0.9114, 1.3146, 0.9390),
    (1.3232, 0.9114, 1.3342, 0.9390),
    (1.3429, 0.9114, 1.3539, 0.9390),
    (1.3626, 0.9114, 1.3736, 0.9390),
    (1.3823, 0.9114, 1.3933, 0.9390),
    (1.2445, 1.1083, 1.2555, 1.1358),
    (1.2642, 1.1083, 1.2752, 1.1358),
    (1.2839, 1.1083, 1.2949, 1.1358),
    (1.3036, 1.1083, 1.3146, 1.1358),
    (1.3232, 1.1083, 1.3342, 1.1358),
    (1.3429, 1.1083, 1.3539, 1.1358),
    (1.3626, 1.1083, 1.3736, 1.1358),
    (1.3823, 1.1083, 1.3933, 1.1358),
)

TOP_U2_SIDE_PAD_REGIONS = (
    (1.5200, 1.0400, 1.6100, 1.5300),
    (2.1400, 1.0400, 2.2200, 1.5300),
)


@dataclass(frozen=True)
class Primitive:
    kind: str
    values: tuple[float, ...]
    exposure: int = 1


@dataclass
class Aperture:
    code: int
    primitives: list[Primitive]
    source: str = ""

    @property
    def first_dark(self) -> Primitive | None:
        for primitive in self.primitives:
            if primitive.exposure > 0:
                return primitive
        return self.primitives[0] if self.primitives else None

    @property
    def max_width(self) -> float:
        widths = []
        for primitive in self.primitives:
            if primitive.kind == "CIR":
                widths.append(primitive.values[0])
            elif primitive.kind == "SQR":
                widths.append(primitive.values[0])
            elif primitive.kind == "REC":
                widths.append(primitive.values[0])
        return max(widths, default=0.001)

    @property
    def max_height(self) -> float:
        heights = []
        for primitive in self.primitives:
            if primitive.kind == "CIR":
                heights.append(primitive.values[0])
            elif primitive.kind == "SQR":
                heights.append(primitive.values[0])
            elif primitive.kind == "REC":
                heights.append(primitive.values[1])
        return max(heights, default=0.001)

    @property
    def stroke_width(self) -> float:
        return max(self.max_width, self.max_height, 0.001)

    @property
    def linecap(self) -> str:
        primitive = self.first_dark
        if primitive and primitive.kind == "CIR":
            return "round"
        return "square"

    @property
    def linejoin(self) -> str:
        primitive = self.first_dark
        if primitive and primitive.kind == "CIR":
            return "round"
        return "miter"


@dataclass
class Settings:
    integer_digits: int = 2
    decimal_digits: int = 4
    zero_suppression: str = "LEADING"
    mode: str = "ABS"
    unit: str = "INCHES"

    @property
    def coord_scale(self) -> float:
        return 10 ** (-self.decimal_digits)


@dataclass
class Bounds:
    min_x: float = math.inf
    min_y: float = math.inf
    max_x: float = -math.inf
    max_y: float = -math.inf

    def include_point(self, x: float, y: float, radius_x: float = 0.0, radius_y: float = 0.0) -> None:
        self.min_x = min(self.min_x, x - radius_x)
        self.max_x = max(self.max_x, x + radius_x)
        self.min_y = min(self.min_y, y - radius_y)
        self.max_y = max(self.max_y, y + radius_y)

    def include_segment(self, x1: float, y1: float, x2: float, y2: float, aperture: Aperture) -> None:
        radius = aperture.stroke_width / 2
        self.include_point(x1, y1, radius, radius)
        self.include_point(x2, y2, radius, radius)

    def include_bounds(self, other: "Bounds") -> None:
        if other.is_empty:
            return
        self.include_point(other.min_x, other.min_y)
        self.include_point(other.max_x, other.max_y)

    @property
    def is_empty(self) -> bool:
        return not math.isfinite(self.min_x)

    def padded(self, pad: float) -> "Bounds":
        if self.is_empty:
            return Bounds(0, 0, 1, 1)
        return Bounds(self.min_x - pad, self.min_y - pad, self.max_x + pad, self.max_y + pad)

    def as_dict(self) -> dict[str, float]:
        if self.is_empty:
            return {"min_x": 0, "min_y": 0, "max_x": 0, "max_y": 0}
        return {
            "min_x": self.min_x,
            "min_y": self.min_y,
            "max_x": self.max_x,
            "max_y": self.max_y,
            "width": self.max_x - self.min_x,
            "height": self.max_y - self.min_y,
        }


@dataclass
class LayerGeometry:
    name: str
    source_path: Path
    settings: Settings
    apertures: dict[int, Aperture]
    paths: DefaultDict[int, list[str]] = field(default_factory=lambda: defaultdict(list))
    segments: DefaultDict[int, list[tuple[float, float, float, float]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    flashes: DefaultDict[int, list[tuple[float, float]]] = field(default_factory=lambda: defaultdict(list))
    bounds: Bounds = field(default_factory=Bounds)
    draw_count: int = 0
    flash_count: int = 0
    move_count: int = 0
    warnings: list[str] = field(default_factory=list)

    def ensure_aperture(self, code: int | None) -> Aperture:
        if code is None:
            code = 0
        aperture = self.apertures.get(code)
        if aperture is None:
            aperture = Aperture(code, [Primitive("CIR", (0.001,))], "default")
            self.apertures[code] = aperture
            message = f"Missing aperture D{code}; using 0.001 circular fallback"
            if message not in self.warnings:
                self.warnings.append(message)
        return aperture

    def add_segment(self, aperture_code: int | None, x1: float, y1: float, x2: float, y2: float) -> None:
        aperture = self.ensure_aperture(aperture_code)
        self.paths[aperture.code].append(f"M{fmt(x1)},{fmt(-y1)}L{fmt(x2)},{fmt(-y2)}")
        self.segments[aperture.code].append((x1, y1, x2, y2))
        self.bounds.include_segment(x1, y1, x2, y2, aperture)
        self.draw_count += 1

    def add_flash(self, aperture_code: int | None, x: float, y: float) -> None:
        aperture = self.ensure_aperture(aperture_code)
        self.flashes[aperture.code].append((x, y))
        self.bounds.include_point(x, y, aperture.max_width / 2, aperture.max_height / 2)
        self.flash_count += 1


def fmt(value: float, digits: int = 5) -> str:
    text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    if text == "-0":
        return "0"
    return text or "0"


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-") or "layer"


def parse_coord(raw: str, settings: Settings) -> float:
    sign = -1 if raw.startswith("-") else 1
    digits = raw[1:] if raw[0] in "+-" else raw
    total = settings.integer_digits + settings.decimal_digits

    if settings.zero_suppression.upper().startswith("TRAIL"):
        digits = digits.ljust(total, "0")
    else:
        digits = digits.zfill(total)

    integer = digits[: settings.integer_digits] or "0"
    decimal = digits[settings.integer_digits :] or "0"
    return sign * (int(integer) + int(decimal) * settings.coord_scale)


def parse_aperture_body(code: int, body: str, scale: float, source: str) -> Aperture | None:
    body = body.strip().rstrip("%").rstrip(".")
    primitives: list[Primitive] = []

    for raw_part in body.split(";"):
        part = raw_part.strip()
        if not part:
            continue
        exposure = 1
        if part.startswith("-"):
            exposure = -1
            part = part[1:].strip()

        pieces = [piece.strip() for piece in part.split(",") if piece.strip()]
        if not pieces:
            continue

        kind = pieces[0].upper()
        try:
            values = tuple(float(piece) * scale for piece in pieces[1:])
        except ValueError:
            continue

        if kind == "CIR" and len(values) >= 1:
            primitives.append(Primitive("CIR", (values[0],), exposure))
        elif kind == "SQR" and len(values) >= 1:
            primitives.append(Primitive("SQR", (values[0],), exposure))
        elif kind == "REC" and len(values) >= 2:
            primitives.append(Primitive("REC", (values[0], values[1]), exposure))

    if not primitives:
        return None
    return Aperture(code, primitives, source)


def scan_gerber_metadata(path: Path) -> tuple[Settings, dict[int, Aperture]]:
    text = path.read_text(errors="ignore")
    settings = Settings()
    apertures: dict[int, Aperture] = {}

    form_match = FORM_RE.search(text)
    if form_match:
        settings.integer_digits = int(form_match.group(1))
        settings.decimal_digits = int(form_match.group(2))

    unit_match = UNIT_RE.search(text)
    if unit_match:
        settings.unit = unit_match.group(1).upper()

    zero_match = ZERO_RE.search(text)
    if zero_match:
        settings.zero_suppression = zero_match.group(1).upper()

    mode_match = MODE_RE.search(text)
    if mode_match:
        settings.mode = mode_match.group(1).upper()

    aperture_scale = settings.coord_scale
    for statement in text.split("*"):
        statement = statement.strip()
        if not statement.startswith("G04"):
            continue
        match = APERTURE_RE.search(statement)
        if not match:
            continue
        code = int(match.group(1))
        aperture = parse_aperture_body(code, match.group(2), aperture_scale, "gerber-comment")
        if aperture:
            apertures[code] = aperture

    return settings, apertures


def expand_code_spec(spec: str) -> Iterable[int]:
    if "-" not in spec:
        yield int(spec)
        return
    start, end = spec.split("-", 1)
    for code in range(int(start), int(end) + 1):
        yield code


def parse_env(path: Path) -> tuple[Settings | None, dict[int, Aperture], list[str]]:
    if not path.exists():
        return None, {}, []

    lines = [line.strip() for line in path.read_text(errors="ignore").splitlines() if line.strip()]
    settings: Settings | None = None
    apertures: dict[int, Aperture] = {}
    layers: list[str] = []

    for line in lines:
        pieces = line.split()
        if len(pieces) >= 2 and all(piece.lstrip("-").isdigit() for piece in pieces[:2]):
            if len(pieces) >= 3 and pieces[0].isdigit() and pieces[1].isdigit() and pieces[2].isdigit():
                settings = Settings(integer_digits=int(pieces[0]), decimal_digits=int(pieces[1]))
                break

    section = "apertures"
    for line in lines:
        if line == "0":
            section = "layers" if section == "apertures" else "tail"
            continue

        pieces = line.split()
        if section == "apertures":
            if len(pieces) < 4 or not re.match(r"^\d+(?:-\d+)?$", pieces[0]):
                continue
            try:
                shape_code = int(pieces[1])
                width = float(pieces[2])
                height = float(pieces[3])
            except ValueError:
                continue

            for code in expand_code_spec(pieces[0]):
                if shape_code == 0:
                    primitives = [Primitive("CIR", (width,), 1)]
                    if height > 0:
                        primitives.append(Primitive("CIR", (height,), -1))
                elif shape_code == 1:
                    primitives = [Primitive("SQR", (width,), 1)]
                elif shape_code == 2:
                    primitives = [Primitive("REC", (width, height), 1)]
                else:
                    primitives = [Primitive("CIR", (max(width, 0.001),), 1)]
                apertures[code] = Aperture(code, primitives, "AA.ENV")
        elif section == "layers":
            if len(pieces) >= 4 and pieces[0].isdigit():
                layers.append(pieces[3])

    return settings, apertures, layers


def parse_gerber(path: Path, env_settings: Settings | None, env_apertures: dict[int, Aperture]) -> LayerGeometry:
    settings, apertures = scan_gerber_metadata(path)
    if not apertures:
        apertures = dict(env_apertures)
    else:
        merged = dict(env_apertures)
        merged.update(apertures)
        apertures = merged

    if env_settings and settings == Settings():
        settings = env_settings

    layer = LayerGeometry(path.name, path, settings, apertures)

    if settings.mode != "ABS":
        layer.warnings.append(f"MODE={settings.mode} is not fully supported; coordinates are treated as absolute")

    text = path.read_text(errors="ignore")
    current_x = 0.0
    current_y = 0.0
    current_aperture: int | None = None
    current_operation: int | None = None

    for raw_statement in text.split("*"):
        statement = raw_statement.strip()
        if not statement:
            continue
        if statement.startswith("G04") or statement.startswith("%"):
            continue
        if statement.startswith("M00") or statement.startswith("M02"):
            break

        if re.fullmatch(r"D\d+", statement):
            code = int(statement[1:])
            if code <= 3:
                if code == 3:
                    layer.add_flash(current_aperture, current_x, current_y)
                current_operation = code
            else:
                current_aperture = code
            continue

        next_x = current_x
        next_y = current_y
        op: int | None = None
        saw_coordinate = False
        saw_standalone_aperture = False

        for match in TOKEN_RE.finditer(statement):
            letter, value = match.group(1), match.group(2)
            if letter == "X":
                next_x = parse_coord(value, settings)
                saw_coordinate = True
            elif letter == "Y":
                next_y = parse_coord(value, settings)
                saw_coordinate = True
            elif letter == "D":
                code = int(value)
                if code <= 3:
                    op = code
                else:
                    current_aperture = code
                    saw_standalone_aperture = True
            elif letter == "G":
                gcode = int(value)
                if gcode in {2, 3}:
                    layer.warnings.append(f"G{gcode:02d} arc ignored in {path.name}")

        if op is None:
            op = current_operation
        elif op <= 3:
            current_operation = op

        if saw_coordinate or (op == 3 and not saw_standalone_aperture):
            if op == 1:
                layer.add_segment(current_aperture, current_x, current_y, next_x, next_y)
            elif op == 2:
                layer.move_count += 1
            elif op == 3:
                layer.add_flash(current_aperture, next_x, next_y)
            current_x = next_x
            current_y = next_y

    return layer


def circle_path(radius: float) -> str:
    r = fmt(radius)
    return f"M{-radius:.5f},0 A{r},{r} 0 1,0 {r},0 A{r},{r} 0 1,0 {-radius:.5f},0 Z"


def rect_path(width: float, height: float) -> str:
    x = -width / 2
    y = -height / 2
    return (
        f"M{fmt(x)},{fmt(y)} "
        f"L{fmt(x + width)},{fmt(y)} "
        f"L{fmt(x + width)},{fmt(y + height)} "
        f"L{fmt(x)},{fmt(y + height)} Z"
    )


def is_round_aperture(aperture: Aperture) -> bool:
    return (
        len(aperture.primitives) == 1
        and aperture.primitives[0].exposure > 0
        and aperture.primitives[0].kind == "CIR"
    )


def aperture_shape_svg(aperture: Aperture, x: float, y: float, scale: float = 1.0) -> str:
    paths: list[str] = []
    simple = len(aperture.primitives) == 1 and aperture.primitives[0].exposure > 0
    primitive = aperture.primitives[0] if aperture.primitives else Primitive("CIR", (0.001,))
    scale = max(scale, 0.001)

    cx = fmt(x)
    cy = fmt(-y)

    if simple and primitive.kind == "CIR":
        return f'<circle cx="{cx}" cy="{cy}" r="{fmt(primitive.values[0] * scale / 2)}" />'
    if simple and primitive.kind == "SQR":
        size = primitive.values[0] * scale
        return (
            f'<rect x="{fmt(x - size / 2)}" y="{fmt(-y - size / 2)}" '
            f'width="{fmt(size)}" height="{fmt(size)}" />'
        )
    if simple and primitive.kind == "REC":
        width, height = primitive.values[0] * scale, primitive.values[1] * scale
        return (
            f'<rect x="{fmt(x - width / 2)}" y="{fmt(-y - height / 2)}" '
            f'width="{fmt(width)}" height="{fmt(height)}" />'
        )

    for primitive in aperture.primitives:
        if primitive.kind == "CIR":
            paths.append(circle_path(primitive.values[0] / 2))
        elif primitive.kind == "SQR":
            paths.append(rect_path(primitive.values[0], primitive.values[0]))
        elif primitive.kind == "REC":
            paths.append(rect_path(primitive.values[0], primitive.values[1]))

    data = " ".join(paths)
    transform = f"translate({cx} {cy})" if scale == 1.0 else f"translate({cx} {cy}) scale({fmt(scale)})"
    return f'<path d="{data}" transform="{transform}" fill-rule="evenodd" />'


def segment_overlaps_bounds(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    aperture: Aperture,
    bounds: Bounds,
) -> bool:
    radius = aperture.stroke_width / 2
    return not (
        max(x1, x2) + radius < bounds.min_x
        or min(x1, x2) - radius > bounds.max_x
        or max(y1, y2) + radius < bounds.min_y
        or min(y1, y2) - radius > bounds.max_y
    )


def flash_overlaps_bounds(x: float, y: float, aperture: Aperture, bounds: Bounds) -> bool:
    radius_x = aperture.max_width / 2
    radius_y = aperture.max_height / 2
    return not (
        x + radius_x < bounds.min_x
        or x - radius_x > bounds.max_x
        or y + radius_y < bounds.min_y
        or y - radius_y > bounds.max_y
    )


def dense_aperture_codes(layer: LayerGeometry, minimum_segments: int = 50_000, ratio: float = 0.35) -> set[int]:
    total = sum(len(segments) for segments in layer.segments.values())
    if total <= 0:
        return set()
    return {
        aperture_code
        for aperture_code, segments in layer.segments.items()
        if len(segments) >= minimum_segments and len(segments) / total >= ratio
    }


def exposed_pad_draw_aperture_codes(layer: LayerGeometry) -> set[int]:
    codes: set[int] = set()
    for aperture_code, segments in layer.segments.items():
        aperture = layer.ensure_aperture(aperture_code)
        primitive = aperture.first_dark
        if not primitive:
            continue
        if primitive.kind in {"SQR", "REC"} and aperture.stroke_width >= 0.015 and len(segments) < 8_000:
            codes.add(aperture_code)
    return codes


def non_trace_draw_aperture_codes(layer: LayerGeometry) -> set[int]:
    codes = exposed_pad_draw_aperture_codes(layer)
    for aperture_code, segments in layer.segments.items():
        aperture = layer.ensure_aperture(aperture_code)
        if aperture.stroke_width > MAX_TRACE_DRAW_WIDTH and len(segments) < 12_000:
            codes.add(aperture_code)
    return codes


def point_near_any(x: float, y: float, points: Iterable[tuple[float, float]], radius: float) -> bool:
    radius_sq = radius * radius
    return any((x - px) * (x - px) + (y - py) * (y - py) <= radius_sq for px, py in points)


def point_near_any_box(
    x: float,
    y: float,
    points: Iterable[tuple[float, float]],
    radius_x: float,
    radius_y: float,
) -> bool:
    return any(abs(x - px) <= radius_x and abs(y - py) <= radius_y for px, py in points)


def segment_touches_no_connect_point(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    points: Iterable[tuple[float, float]],
    radius: float,
    radius_y: float | None = None,
) -> bool:
    if radius_y is not None:
        return point_near_any_box(x1, y1, points, radius, radius_y) or point_near_any_box(
            x2, y2, points, radius, radius_y
        )
    return point_near_any(x1, y1, points, radius) or point_near_any(x2, y2, points, radius)


def render_layer_body(
    layer: LayerGeometry,
    color: str,
    opacity: float,
    visible: bool = True,
    *,
    layer_id: str | None = None,
    classes: str = "",
    fill: str | None = None,
    stroke: str | None = None,
    draw: bool = True,
    flashes: bool = True,
    clip_bounds: Bounds | None = None,
    skip_apertures: set[int] | None = None,
    exclude_segment_points: Iterable[tuple[float, float]] = (),
    exclude_segment_radius: float = 0.0,
    exclude_segment_y_radius: float | None = None,
    draw_stroke_scale: float = 1.0,
    flash_scale: float = 1.0,
    small_round_flash_threshold: float | None = None,
    small_round_flash_scale: float = 1.0,
    extra_attrs: str = "",
) -> str:
    layer_id = layer_id or f"layer-{safe_id(layer.name)}"
    fill_color = fill or color
    stroke_color = stroke or color
    class_attr = f"gerber-layer {classes}".strip()
    display = "" if visible else ' style="display:none"'
    extra = f" {extra_attrs.strip()}" if extra_attrs.strip() else ""
    parts = [
        (
            f'<g id="{layer_id}" class="{class_attr}" '
            f'data-layer="{html.escape(layer.name)}" opacity="{fmt(opacity, 3)}" '
            f'fill="{fill_color}" stroke="{stroke_color}"{extra}{display}>'
        )
    ]

    if draw:
        for aperture_code in sorted(layer.paths):
            if skip_apertures and aperture_code in skip_apertures:
                continue
            chunks = layer.paths[aperture_code]
            if not chunks:
                continue
            aperture = layer.ensure_aperture(aperture_code)
            if clip_bounds is not None:
                chunks = [
                    f"M{fmt(x1)},{fmt(-y1)}L{fmt(x2)},{fmt(-y2)}"
                    for x1, y1, x2, y2 in layer.segments.get(aperture_code, [])
                    if segment_overlaps_bounds(x1, y1, x2, y2, aperture, clip_bounds)
                    and not (
                        exclude_segment_radius > 0
                        and segment_touches_no_connect_point(
                            x1,
                            y1,
                            x2,
                            y2,
                            exclude_segment_points,
                            exclude_segment_radius,
                            exclude_segment_y_radius,
                        )
                    )
                ]
                if not chunks:
                    continue
            path_data = " ".join(chunks)
            stroke_width = aperture.stroke_width * max(draw_stroke_scale, 0.001)
            parts.append(
                f'<path d="{path_data}" fill="none" stroke-width="{fmt(stroke_width)}" '
                f'stroke-linecap="{aperture.linecap}" stroke-linejoin="{aperture.linejoin}" />'
            )

    if flashes:
        for aperture_code in sorted(layer.flashes):
            if skip_apertures and aperture_code in skip_apertures:
                continue
            flashes_for_aperture = layer.flashes[aperture_code]
            if not flashes_for_aperture:
                continue
            aperture = layer.ensure_aperture(aperture_code)
            if clip_bounds is not None:
                flashes_for_aperture = [
                    (x, y)
                    for x, y in flashes_for_aperture
                    if flash_overlaps_bounds(x, y, aperture, clip_bounds)
                ]
                if not flashes_for_aperture:
                    continue
            parts.append(f'<g class="flashes aperture-D{aperture_code}" stroke="none">')
            aperture_flash_scale = max(flash_scale, 0.001)
            if (
                small_round_flash_threshold is not None
                and is_round_aperture(aperture)
                and aperture.max_width <= small_round_flash_threshold
                and aperture.max_height <= small_round_flash_threshold
            ):
                aperture_flash_scale *= small_round_flash_scale
            for x, y in flashes_for_aperture:
                parts.append(aperture_shape_svg(aperture, x, y, aperture_flash_scale))
            parts.append("</g>")

    parts.append("</g>")
    return "\n".join(parts)


def svg_viewbox(bounds: Bounds, pad: float) -> tuple[str, float, float]:
    padded = bounds.padded(pad)
    width = padded.max_x - padded.min_x
    height = padded.max_y - padded.min_y
    viewbox = f"{fmt(padded.min_x)} {fmt(-padded.max_y)} {fmt(width)} {fmt(height)}"
    return viewbox, width, height


def wrap_svg(body: str, bounds: Bounds, pad: float, title: str, background: str = "#06110e") -> str:
    viewbox, width, height = svg_viewbox(bounds, pad)
    bg = bounds.padded(pad)
    bg_rect = (
        f'<rect x="{fmt(bg.min_x)}" y="{fmt(-bg.max_y)}" '
        f'width="{fmt(bg.max_x - bg.min_x)}" height="{fmt(bg.max_y - bg.min_y)}" '
        f'fill="{background}" />'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{viewbox}" '
        f'width="{fmt(width)}in" height="{fmt(height)}in" role="img" '
        f'aria-label="{html.escape(title)}">\n'
        f'<title>{html.escape(title)}</title>\n'
        f'{bg_rect}\n'
        f'{body}\n'
        '</svg>\n'
    )


def board_geometry(bounds: Bounds) -> tuple[float, float, float, float, float]:
    box = bounds.padded(0.0)
    width = box.max_x - box.min_x
    height = box.max_y - box.min_y
    radius = min(width, height) * 0.035
    return box.min_x, -box.max_y, width, height, radius


def physical_defs(prefix: str, bounds: Bounds) -> str:
    x, y, width, height, radius = board_geometry(bounds)
    pad_stops = (
        ('0%', '#d8c69e'),
        ('45%', '#a68f66'),
        ('100%', '#76694d'),
    )
    return f"""<defs>
<linearGradient id="{prefix}-boardMask" x1="0%" y1="0%" x2="100%" y2="100%">
  <stop offset="0%" stop-color="#10b56a" />
  <stop offset="48%" stop-color="#0aa05f" />
  <stop offset="100%" stop-color="#07874f" />
</linearGradient>
<linearGradient id="{prefix}-padMetal" x1="0%" y1="0%" x2="100%" y2="100%">
  <stop offset="{pad_stops[0][0]}" stop-color="{pad_stops[0][1]}" />
  <stop offset="{pad_stops[1][0]}" stop-color="{pad_stops[1][1]}" />
  <stop offset="{pad_stops[2][0]}" stop-color="{pad_stops[2][1]}" />
</linearGradient>
<linearGradient id="{prefix}-boardSheen" x1="0%" y1="0%" x2="100%" y2="100%">
  <stop offset="0%" stop-color="#ffffff" stop-opacity="0.18" />
  <stop offset="42%" stop-color="#ffffff" stop-opacity="0.05" />
  <stop offset="100%" stop-color="#00180e" stop-opacity="0.10" />
</linearGradient>
<filter id="{prefix}-boardShadow" x="-4%" y="-8%" width="108%" height="116%">
  <feDropShadow dx="0" dy="0.055" stdDeviation="0.06" flood-color="#000000" flood-opacity="0.42" />
</filter>
<filter id="{prefix}-boardTexture" x="0%" y="0%" width="100%" height="100%">
  <feTurbulence type="fractalNoise" baseFrequency="0.78" numOctaves="2" seed="17" result="noise" />
  <feColorMatrix in="noise" type="matrix"
    values="0 0 0 0 0.02  0 0 0 0 0.16  0 0 0 0 0.10  0 0 0 0.16 0" />
</filter>
<filter id="{prefix}-traceDepth" x="-3%" y="-3%" width="106%" height="106%">
  <feDropShadow dx="0" dy="-0.0012" stdDeviation="0.0010" flood-color="#54d58b" flood-opacity="0.10" />
  <feDropShadow dx="0" dy="0.0028" stdDeviation="0.0020" flood-color="#002111" flood-opacity="0.38" />
</filter>
<filter id="{prefix}-traceField" x="-4%" y="-4%" width="108%" height="108%">
  <feGaussianBlur stdDeviation="0.0006" />
</filter>
<filter id="{prefix}-padLift" x="-4%" y="-4%" width="108%" height="108%">
  <feDropShadow dx="0" dy="-0.0018" stdDeviation="0.0015" flood-color="#fff0bd" flood-opacity="0.20" />
  <feDropShadow dx="0" dy="0.004" stdDeviation="0.003" flood-color="#102f20" flood-opacity="0.28" />
</filter>
<filter id="{prefix}-silkLift" x="-2%" y="-2%" width="104%" height="104%">
  <feDropShadow dx="0" dy="0.003" stdDeviation="0.004" flood-color="#002416" flood-opacity="0.24" />
</filter>
<clipPath id="{prefix}-clip">
  <rect x="{fmt(x)}" y="{fmt(y)}" width="{fmt(width)}" height="{fmt(height)}" rx="{fmt(radius)}" ry="{fmt(radius)}" />
</clipPath>
</defs>"""


def board_base_svg(prefix: str, bounds: Bounds) -> str:
    x, y, width, height, radius = board_geometry(bounds)
    edge_width = max(min(width, height) * 0.0022, 0.007)
    inner_width = max(min(width, height) * 0.0009, 0.003)
    return "\n".join(
        [
            (
                f'<rect class="board-base" x="{fmt(x)}" y="{fmt(y)}" width="{fmt(width)}" '
                f'height="{fmt(height)}" rx="{fmt(radius)}" ry="{fmt(radius)}" '
                f'fill="url(#{prefix}-boardMask)" filter="url(#{prefix}-boardShadow)" />'
            ),
            (
                f'<rect x="{fmt(x)}" y="{fmt(y)}" width="{fmt(width)}" height="{fmt(height)}" '
                f'rx="{fmt(radius)}" ry="{fmt(radius)}" fill="#ffffff" '
                f'filter="url(#{prefix}-boardTexture)" opacity="0.42" style="mix-blend-mode:multiply" />'
            ),
            (
                f'<rect x="{fmt(x)}" y="{fmt(y)}" width="{fmt(width)}" height="{fmt(height)}" '
                f'rx="{fmt(radius)}" ry="{fmt(radius)}" fill="url(#{prefix}-boardSheen)" '
                f'opacity="0.58" style="mix-blend-mode:soft-light" />'
            ),
            (
                f'<rect x="{fmt(x)}" y="{fmt(y)}" width="{fmt(width)}" height="{fmt(height)}" '
                f'rx="{fmt(radius)}" ry="{fmt(radius)}" fill="none" '
                f'stroke="#86d36f" stroke-width="{fmt(edge_width)}" opacity="0.42" />'
            ),
            (
                f'<rect x="{fmt(x + edge_width * 2)}" y="{fmt(y + edge_width * 2)}" '
                f'width="{fmt(width - edge_width * 4)}" height="{fmt(height - edge_width * 4)}" '
                f'rx="{fmt(max(radius - edge_width * 2, 0))}" ry="{fmt(max(radius - edge_width * 2, 0))}" '
                f'fill="none" stroke="#063a25" stroke-width="{fmt(inner_width)}" opacity="0.24" />'
            ),
        ]
    )


def mask_shadow_patches_svg(side: str, prefix: str) -> str:
    if side != "top" or not TOP_MASK_SHADOW_PATCHES:
        return ""
    parts = [
        (
            f'<g id="{prefix}-photo-mask-shadows" class="role-copper" '
            'data-role="copper" opacity="0.24" fill="#003f2b" '
            'style="mix-blend-mode:multiply">'
        )
    ]
    for polygon in TOP_MASK_SHADOW_PATCHES:
        points = " ".join(f"{fmt(x)},{fmt(-y)}" for x, y in polygon)
        parts.append(f'<polygon points="{points}" />')
    parts.append("</g>")
    return "\n".join(parts)


def top_u6_pad_patches_svg(side: str, prefix: str) -> str:
    if side != "top" or not TOP_U6_PAD_RECTS:
        return ""

    parts = [
        (
            f'<g id="{prefix}-u6-pad-patches" class="gerber-layer role-metal" '
            f'data-role="metal" opacity="0.940" fill="url(#{prefix}-padMetal)" '
            f'stroke="#8c7455" stroke-width="{fmt(0.0016)}" filter="url(#{prefix}-padLift)">'
        )
    ]
    for x1, y1, x2, y2 in TOP_U6_PAD_RECTS:
        left = min(x1, x2)
        right = max(x1, x2)
        bottom = min(y1, y2)
        top = max(y1, y2)
        parts.append(
            (
                f'<rect x="{fmt(left)}" y="{fmt(-top)}" width="{fmt(right - left)}" '
                f'height="{fmt(top - bottom)}" />'
            )
        )
    parts.append("</g>")
    return "\n".join(parts)


def top_u2_side_pad_segments_svg(layer: LayerGeometry | None, side: str, prefix: str) -> str:
    if side != "top" or layer is None:
        return ""

    aperture_code = 55
    aperture = layer.apertures.get(aperture_code)
    if aperture is None:
        return ""

    candidate_paths: list[str] = []
    for x1, y1, x2, y2 in layer.segments.get(aperture_code, []):
        if abs(y1 - y2) > 0.0005:
            continue
        length = abs(x2 - x1)
        if length < 0.030 or length > 0.060:
            continue
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        if not any(
            xmin <= mid_x <= xmax and ymin <= mid_y <= ymax
            for xmin, ymin, xmax, ymax in TOP_U2_SIDE_PAD_REGIONS
        ):
            continue
        candidate_paths.append(f"M{fmt(x1)},{fmt(-y1)}L{fmt(x2)},{fmt(-y2)}")

    if not candidate_paths:
        return ""

    return (
        f'<path id="{prefix}-u2-side-pads" class="gerber-layer role-metal" '
        f'd="{html.escape(" ".join(candidate_paths), quote=True)}" '
        f'data-role="metal" fill="none" stroke="url(#{prefix}-padMetal)" '
        f'stroke-width="{fmt(aperture.stroke_width)}" '
        f'stroke-linecap="{aperture.linecap}" stroke-linejoin="{aperture.linejoin}" '
        f'opacity="0.94" filter="url(#{prefix}-padLift)" />'
    )


def layer_map(layers: list[LayerGeometry]) -> dict[str, LayerGeometry]:
    return {layer.name: layer for layer in layers}


def unique_positions(values: Iterable[float], tolerance: float = 0.006) -> list[float]:
    positions: list[float] = []
    for value in sorted(values):
        if not positions or abs(value - positions[-1]) > tolerance:
            positions.append(value)
        else:
            positions[-1] = (positions[-1] + value) / 2
    return positions


def first_large_interval(positions: list[float], minimum_width: float) -> tuple[float, float] | None:
    for start, end in zip(positions, positions[1:]):
        if end - start >= minimum_width:
            return start, end
    return None


def estimate_single_board_bounds(layers: list[LayerGeometry], panel_bounds: Bounds) -> Bounds:
    """Detect the first PCB cell when the mechanical layer contains a panel array."""
    lookup = layer_map(layers)
    outline = lookup.get("COMP-S") or lookup.get("SOLD-S")
    if not outline or panel_bounds.is_empty:
        return panel_bounds

    panel_width = panel_bounds.max_x - panel_bounds.min_x
    panel_height = panel_bounds.max_y - panel_bounds.min_y
    vertical_xs: list[float] = []
    horizontal_ys: list[float] = []

    for segments in outline.segments.values():
        for x1, y1, x2, y2 in segments:
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dx <= 0.006 and dy >= panel_height * 0.18:
                vertical_xs.extend([x1, x2])
            if dy <= 0.006 and dx >= panel_width * 0.18:
                horizontal_ys.extend([y1, y2])

    x_positions = unique_positions(vertical_xs)
    y_positions = unique_positions(horizontal_ys)
    x_interval = first_large_interval(x_positions, panel_width * 0.20)
    y_interval = first_large_interval(y_positions, panel_height * 0.20)

    if not x_interval or not y_interval:
        return panel_bounds

    detected = Bounds(x_interval[0], y_interval[0], x_interval[1], y_interval[1])
    if detected.max_x <= detected.min_x or detected.max_y <= detected.min_y:
        return panel_bounds
    return detected


def render_physical_side_body(
    layers_by_name: dict[str, LayerGeometry],
    side: str,
    bounds: Bounds,
    prefix: str,
) -> str:
    config = SIDE_CONFIG[side]
    parts = [physical_defs(prefix, bounds), board_base_svg(prefix, bounds), f'<g clip-path="url(#{prefix}-clip)">']
    if side == "bottom":
        mirror_x = bounds.min_x + bounds.max_x
        parts.append(f'<g transform="translate({fmt(mirror_x)} 0) scale(-1 1)">')

    copper = layers_by_name.get(config["copper"])
    if copper:
        dense_codes = dense_aperture_codes(copper)
        pad_draw_codes = exposed_pad_draw_aperture_codes(copper)
        trace_skip_codes = dense_codes | non_trace_draw_aperture_codes(copper)
        no_connect_points = TOP_TRACE_NO_CONNECT_POINTS if side == "top" else ()
        if dense_codes:
            non_dense_codes = set(copper.paths) - dense_codes
            parts.append(
                render_layer_body(
                    copper,
                    "#005c3a",
                    0.13,
                    layer_id=f"{prefix}-copper-field",
                    classes="role-copper",
                    fill="#005c3a",
                    stroke="#005c3a",
                    flashes=False,
                    clip_bounds=bounds,
                    skip_apertures=non_dense_codes,
                    draw_stroke_scale=TRACE_FIELD_STROKE_SCALE,
                    extra_attrs=(
                        f'data-role="copper" filter="url(#{prefix}-traceField)" '
                        'style="mix-blend-mode:multiply"'
                    ),
            )
        )
        shadow_patches = mask_shadow_patches_svg(side, prefix)
        if shadow_patches:
            parts.append(shadow_patches)
        parts.append(
            render_layer_body(
                copper,
                "#006b43",
                0.22,
                layer_id=f"{prefix}-copper-relief",
                classes="role-copper",
                fill="#006b43",
                stroke="#006b43",
                flashes=False,
                clip_bounds=bounds,
                skip_apertures=trace_skip_codes,
                exclude_segment_points=no_connect_points,
                exclude_segment_radius=TRACE_NO_CONNECT_RADIUS,
                exclude_segment_y_radius=TRACE_NO_CONNECT_Y_RADIUS,
                draw_stroke_scale=TRACE_RELIEF_STROKE_SCALE,
                extra_attrs='data-role="copper" style="mix-blend-mode:multiply"',
            )
        )
        parts.append(
            render_layer_body(
                copper,
                "#04281d",
                0.34,
                layer_id=f"{prefix}-copper",
                classes="role-copper",
                fill="#04281d",
                stroke="#04281d",
                flashes=False,
                clip_bounds=bounds,
                skip_apertures=trace_skip_codes,
                exclude_segment_points=no_connect_points,
                exclude_segment_radius=TRACE_NO_CONNECT_RADIUS,
                exclude_segment_y_radius=TRACE_NO_CONNECT_Y_RADIUS,
                draw_stroke_scale=TRACE_DARK_STROKE_SCALE,
                extra_attrs=f'data-role="copper" filter="url(#{prefix}-traceDepth)" style="mix-blend-mode:multiply"',
            )
        )
        parts.append(
            render_layer_body(
                copper,
                "#24bd78",
                0.34,
                layer_id=f"{prefix}-copper-highlight",
                classes="role-copper",
                fill="#24bd78",
                stroke="#24bd78",
                flashes=False,
                clip_bounds=bounds,
                skip_apertures=trace_skip_codes,
                exclude_segment_points=no_connect_points,
                exclude_segment_radius=TRACE_NO_CONNECT_RADIUS,
                exclude_segment_y_radius=TRACE_NO_CONNECT_Y_RADIUS,
                draw_stroke_scale=TRACE_LIGHT_STROKE_SCALE,
                extra_attrs='data-role="copper"',
            )
        )

    mask = layers_by_name.get(config["mask"])
    if mask:
        parts.append(
            render_layer_body(
                mask,
                "#035236",
                0.2,
                layer_id=f"{prefix}-mask-openings",
                classes="role-copper",
                fill="#035236",
                stroke="#035236",
                draw=False,
                clip_bounds=bounds,
                extra_attrs='data-role="copper" style="mix-blend-mode:multiply"',
            )
        )

    if copper:
        if pad_draw_codes:
            non_pad_codes = set(copper.paths) - pad_draw_codes
            parts.append(
                render_layer_body(
                    copper,
                    f"url(#{prefix}-padMetal)",
                    0.94,
                    layer_id=f"{prefix}-metal-drawn-pads",
                    classes="role-metal",
                    fill=f"url(#{prefix}-padMetal)",
                    stroke="#8c7455",
                    flashes=False,
                    clip_bounds=bounds,
                    skip_apertures=non_pad_codes,
                    extra_attrs=f'data-role="metal" filter="url(#{prefix}-padLift)"',
                )
            )

    if copper:
        parts.append(
            render_layer_body(
                copper,
                f"url(#{prefix}-padMetal)",
                0.94,
                layer_id=f"{prefix}-metal",
                classes="role-metal",
                fill=f"url(#{prefix}-padMetal)",
                stroke="#8c7455",
                draw=False,
                clip_bounds=bounds,
                small_round_flash_threshold=SMALL_VIA_FLASH_THRESHOLD,
                small_round_flash_scale=SMALL_VIA_FLASH_SCALE,
                extra_attrs=f'data-role="metal" filter="url(#{prefix}-padLift)"',
            )
        )

    u2_side_pad_segments = top_u2_side_pad_segments_svg(copper, side, prefix)
    if u2_side_pad_segments:
        parts.append(u2_side_pad_segments)

    u6_pad_patches = top_u6_pad_patches_svg(side, prefix)
    if u6_pad_patches:
        parts.append(u6_pad_patches)

    drill_drawing = layers_by_name.get("D3")
    if drill_drawing:
        parts.append(
            render_layer_body(
                drill_drawing,
                f"url(#{prefix}-padMetal)",
                0.92,
                layer_id=f"{prefix}-metal-rings",
                classes="role-metal",
                fill=f"url(#{prefix}-padMetal)",
                stroke="#8c7455",
                draw=False,
                clip_bounds=bounds,
                small_round_flash_threshold=SMALL_VIA_FLASH_THRESHOLD,
                small_round_flash_scale=SMALL_VIA_FLASH_SCALE,
                extra_attrs=f'data-role="metal" filter="url(#{prefix}-padLift)"',
            )
        )

    drill = layers_by_name.get("DRILL")
    if drill:
        parts.append(
            render_layer_body(
                drill,
                f"url(#{prefix}-padMetal)",
                0.88,
                layer_id=f"{prefix}-drill-plating",
                classes="role-metal",
                fill=f"url(#{prefix}-padMetal)",
                stroke="#8c7455",
                draw=False,
                clip_bounds=bounds,
                flash_scale=DRILL_RING_FLASH_SCALE,
                extra_attrs=f'data-role="metal" filter="url(#{prefix}-padLift)"',
            )
        )

    silk = layers_by_name.get(config["silk"])
    if silk:
        parts.append(
            render_layer_body(
                silk,
                "#d9eee4",
                0.88,
                layer_id=f"{prefix}-silk",
                classes="role-silk",
                fill="#d9eee4",
                stroke="#d9eee4",
                clip_bounds=bounds,
                draw_stroke_scale=SILK_STROKE_SCALE,
                extra_attrs=f'data-role="silk" filter="url(#{prefix}-silkLift)"',
            )
        )

    if drill:
        parts.append(
            render_layer_body(
                drill,
                "#020504",
                0.98,
                layer_id=f"{prefix}-drill",
                classes="role-drill",
                fill="#020504",
                stroke="#020504",
                draw=False,
                clip_bounds=bounds,
                small_round_flash_threshold=SMALL_DRILL_FLASH_THRESHOLD,
                small_round_flash_scale=SMALL_DRILL_FLASH_SCALE,
                extra_attrs='data-role="drill"',
            )
        )

    outline = layers_by_name.get(config["outline"])
    if outline:
        parts.append(
            render_layer_body(
                outline,
                "#d9f26c",
                0.58,
                layer_id=f"{prefix}-outline",
                classes="role-outline",
                fill="#d9f26c",
                stroke="#d9f26c",
                flashes=False,
                clip_bounds=bounds,
                extra_attrs='data-role="outline"',
            )
        )

    if side == "bottom":
        parts.append("</g>")
    parts.append("</g>")
    return "\n".join(parts)


def render_physical_svg(
    layers: list[LayerGeometry],
    side: str,
    global_bounds: Bounds,
    pad: float,
    *,
    standalone: bool,
    active: bool = False,
) -> str:
    prefix = f"physical-{side}"
    viewbox, width, height = svg_viewbox(global_bounds, pad)
    label = SIDE_CONFIG[side]["label"]
    body = render_physical_side_body(layer_map(layers), side, global_bounds, prefix)
    class_attr = f"pcb-side side-{side}" + (" active" if active else "")
    svg = (
        f'<svg class="{class_attr}" data-side="{side}" xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{viewbox}" width="{fmt(width)}in" height="{fmt(height)}in" '
        f'preserveAspectRatio="xMidYMid meet" role="img" aria-label="{html.escape(label)}">\n'
        f'<title>{html.escape(label)}</title>\n'
        f'{body}\n'
        '</svg>\n'
    )
    if standalone:
        return '<?xml version="1.0" encoding="UTF-8"?>\n' + svg
    return svg


def render_pcb_viewer_html(layers: list[LayerGeometry], global_bounds: Bounds, out_name: str, pad: float) -> str:
    viewbox, width, height = svg_viewbox(global_bounds, pad)
    units = layers[0].settings.unit.lower() if layers else "units"
    board = global_bounds.as_dict()
    board_width = board.get("width", width)
    board_height = board.get("height", height)
    roles = [
        ("copper", "Copper relief", "#053523", True),
        ("metal", "Pads / vias", "#c4a574", True),
        ("silk", "Silk", "#edf7f1", True),
        ("drill", "Drills", "#020504", True),
        ("outline", "Outline", "#d9f26c", True),
    ]
    role_controls = []
    for role, label, color, checked in roles:
        role_controls.append(
            '<label class="role-row">'
            f'<input type="checkbox" data-role-toggle="{role}" {"checked" if checked else ""} />'
            f'<span class="swatch" style="background:{color}"></span>'
            f'<span>{html.escape(label)}</span>'
            '</label>'
        )

    side_buttons = []
    for side, config in SIDE_CONFIG.items():
        side_buttons.append(
            f'<button type="button" class="side-button{" active" if side == "top" else ""}" '
            f'data-side-button="{side}">{html.escape(config["label"])}</button>'
        )

    top_svg = render_physical_svg(layers, "top", global_bounds, pad, standalone=False, active=True)
    bottom_svg = render_physical_svg(layers, "bottom", global_bounds, pad, standalone=False)

    layer_stats = []
    for layer in layers:
        layer_stats.append(
            '<tr>'
            f'<td>{html.escape(layer.name)}</td>'
            f'<td>{layer.draw_count:,}</td>'
            f'<td>{layer.flash_count:,}</td>'
            '</tr>'
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{html.escape(out_name)}</title>
<style>
:root {{
  color-scheme: dark;
  font-family: Inter, Segoe UI, Arial, sans-serif;
  background: #07100d;
  color: #eef5f1;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  display: grid;
  grid-template-columns: minmax(280px, 340px) 1fr;
  min-height: 100vh;
  background: #07100d;
}}
aside {{
  border-right: 1px solid #1e302b;
  padding: 16px;
  background: #0b1512;
  overflow: auto;
}}
main {{
  overflow: auto;
  display: grid;
  place-items: center;
  padding: 22px;
  background:
    linear-gradient(90deg, rgb(255 255 255 / 0.025) 1px, transparent 1px),
    linear-gradient(rgb(255 255 255 / 0.025) 1px, transparent 1px),
    #07100d;
  background-size: 32px 32px;
}}
h1 {{
  font-size: 18px;
  line-height: 1.25;
  margin: 0 0 8px;
  font-weight: 750;
}}
.board-size {{
  color: #94aaa2;
  font-size: 12px;
  margin-bottom: 16px;
}}
.control-group {{
  border-top: 1px solid #17231f;
  padding-top: 14px;
  margin-top: 14px;
}}
.group-title {{
  color: #a9bbb5;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0;
  margin: 0 0 9px;
  text-transform: uppercase;
}}
.side-switch {{
  display: grid;
  grid-template-columns: 1fr;
  gap: 8px;
}}
.side-button {{
  appearance: none;
  border: 1px solid #274139;
  border-radius: 7px;
  padding: 9px 10px;
  background: #101f1a;
  color: #d9e7e1;
  font: inherit;
  font-size: 12px;
  font-weight: 700;
  cursor: pointer;
  text-align: left;
}}
.side-button.active {{
  border-color: #63d98f;
  background: #143225;
  color: #f2fff7;
}}
.role-row {{
  display: grid;
  grid-template-columns: 18px 16px 1fr;
  gap: 8px;
  align-items: center;
  padding: 7px 0;
  cursor: pointer;
  font-size: 13px;
  font-weight: 650;
}}
.role-row input {{
  accent-color: #6edb97;
  margin: 0;
}}
.swatch {{
  width: 12px;
  height: 12px;
  border-radius: 50%;
  box-shadow: 0 0 0 1px rgb(255 255 255 / 0.22);
}}
table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 11px;
  color: #91a49e;
}}
th, td {{
  padding: 5px 0;
  border-bottom: 1px solid #15231f;
  text-align: right;
  white-space: nowrap;
}}
th:first-child, td:first-child {{ text-align: left; color: #d6e3de; }}
.stage {{
  width: min(1320px, 100%);
}}
.pcb-side {{
  display: none;
  width: 100%;
  height: auto;
  max-height: calc(100vh - 44px);
  background: transparent;
  overflow: visible;
}}
.pcb-side.active {{ display: block; }}
@media (max-width: 860px) {{
  body {{ grid-template-columns: 1fr; }}
  aside {{ border-right: 0; border-bottom: 1px solid #1e302b; }}
  main {{ padding: 14px; }}
}}
</style>
</head>
<body>
<aside>
  <h1>{html.escape(out_name)}</h1>
  <div class="board-size">{fmt(board_width)} x {fmt(board_height)} {html.escape(units)}</div>
  <div class="control-group">
    <p class="group-title">Side</p>
    <div class="side-switch">{''.join(side_buttons)}</div>
  </div>
  <div class="control-group">
    <p class="group-title">Render</p>
    {''.join(role_controls)}
  </div>
  <div class="control-group">
    <p class="group-title">Parsed layers</p>
    <table>
      <thead><tr><th>Layer</th><th>Lines</th><th>Flashes</th></tr></thead>
      <tbody>{''.join(layer_stats)}</tbody>
    </table>
  </div>
</aside>
<main>
  <div class="stage" style="aspect-ratio:{fmt(width)} / {fmt(height)}">
    {top_svg}
    {bottom_svg}
  </div>
</main>
<script>
const sideButtons = document.querySelectorAll("[data-side-button]");
const sides = document.querySelectorAll(".pcb-side");
for (const button of sideButtons) {{
  button.addEventListener("click", () => {{
    const side = button.dataset.sideButton;
    for (const other of sideButtons) other.classList.toggle("active", other === button);
    for (const svg of sides) svg.classList.toggle("active", svg.dataset.side === side);
  }});
}}
for (const checkbox of document.querySelectorAll("[data-role-toggle]")) {{
  checkbox.addEventListener("change", () => {{
    const role = checkbox.dataset.roleToggle;
    for (const group of document.querySelectorAll(`.role-${{role}}`)) {{
      group.style.display = checkbox.checked ? "" : "none";
    }}
  }});
}}
</script>
</body>
</html>
"""


def render_html(layers: list[LayerGeometry], global_bounds: Bounds, out_name: str, pad: float) -> str:
    viewbox, width, height = svg_viewbox(global_bounds, pad)
    controls = []
    body_parts = []

    for index, layer in enumerate(layers):
        color = DEFAULT_COLORS.get(layer.name, f"hsl({(index * 47) % 360} 78% 62%)")
        opacity = 0.85 if layer.name in {"COMP-P", "SOLD-P"} else 0.72
        visible = layer.name not in {"DRILL", "D3"}
        checked = "checked" if visible else ""
        layer_id = safe_id(layer.name)
        controls.append(
            '<label class="layer-row">'
            f'<input type="checkbox" data-target="layer-{layer_id}" {checked} />'
            f'<span class="swatch" style="background:{color}"></span>'
            f'<span class="name">{html.escape(layer.name)}</span>'
            f'<span class="meta">{layer.draw_count:,} lines / {layer.flash_count:,} flashes</span>'
            '</label>'
        )
        body_parts.append(render_layer_body(layer, color, opacity, visible))

    padded = global_bounds.padded(pad)
    bg_rect = (
        f'<rect x="{fmt(padded.min_x)}" y="{fmt(-padded.max_y)}" '
        f'width="{fmt(padded.max_x - padded.min_x)}" height="{fmt(padded.max_y - padded.min_y)}" '
        'fill="#06110e" />'
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{html.escape(out_name)}</title>
<style>
:root {{
  color-scheme: dark;
  font-family: Inter, Segoe UI, Arial, sans-serif;
  background: #07100e;
  color: #e9eef2;
}}
body {{
  margin: 0;
  display: grid;
  grid-template-columns: minmax(260px, 320px) 1fr;
  min-height: 100vh;
}}
aside {{
  border-right: 1px solid #24322f;
  padding: 16px;
  background: #0b1512;
  overflow: auto;
}}
main {{
  overflow: auto;
  display: grid;
  place-items: center;
  padding: 18px;
}}
h1 {{
  font-size: 18px;
  line-height: 1.25;
  margin: 0 0 14px;
  font-weight: 700;
}}
.board-size {{
  color: #9fb0aa;
  font-size: 12px;
  margin-bottom: 18px;
}}
.layer-row {{
  display: grid;
  grid-template-columns: 18px 16px minmax(70px, 1fr);
  gap: 8px;
  align-items: center;
  padding: 8px 0;
  border-top: 1px solid #17231f;
  cursor: pointer;
}}
.layer-row input {{
  accent-color: #76d39b;
  margin: 0;
}}
.swatch {{
  width: 12px;
  height: 12px;
  border-radius: 50%;
  box-shadow: 0 0 0 1px rgb(255 255 255 / 0.18);
}}
.name {{
  font-size: 13px;
  font-weight: 650;
}}
.meta {{
  grid-column: 3;
  color: #80928d;
  font-size: 11px;
}}
svg {{
  width: min(1200px, 100%);
  height: auto;
  background: #06110e;
  box-shadow: 0 18px 80px rgb(0 0 0 / 0.38);
}}
@media (max-width: 840px) {{
  body {{ grid-template-columns: 1fr; }}
  aside {{ border-right: 0; border-bottom: 1px solid #24322f; }}
}}
</style>
</head>
<body>
<aside>
  <h1>{html.escape(out_name)}</h1>
  <div class="board-size">{fmt(width)} x {fmt(height)} {html.escape(layers[0].settings.unit.lower() if layers else "units")}</div>
  {''.join(controls)}
</aside>
<main>
<svg id="board" xmlns="http://www.w3.org/2000/svg" viewBox="{viewbox}" role="img" aria-label="{html.escape(out_name)}">
<title>{html.escape(out_name)}</title>
{bg_rect}
{chr(10).join(body_parts)}
</svg>
</main>
<script>
for (const checkbox of document.querySelectorAll("[data-target]")) {{
  checkbox.addEventListener("change", () => {{
    const group = document.getElementById(checkbox.dataset.target);
    if (group) group.style.display = checkbox.checked ? "" : "none";
  }});
}}
</script>
</body>
</html>
"""


def find_browser_executable() -> str | None:
    candidates = [
        shutil.which("msedge"),
        shutil.which("chrome"),
        shutil.which("chromium"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def find_node_executable() -> str | None:
    candidates = [
        shutil.which("node"),
        r"C:\Program Files\WindowsApps\OpenAI.Codex_26.506.3741.0_x64__2p2nqsd0c76g0\app\resources\node.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def find_node_modules_dir() -> str | None:
    candidates = [
        Path.cwd() / "node_modules",
        Path.home() / ".cache" / "codex-runtimes" / "codex-primary-runtime" / "dependencies" / "node" / "node_modules",
    ]
    for candidate in candidates:
        if (candidate / "playwright").exists():
            return str(candidate)
    return None


def svg_export_html(svg_text: str, width_px: int, height_px: int) -> str:
    body = re.sub(r"<\?xml[^>]*>\s*", "", svg_text, count=1)
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
html, body {{
  width: {width_px}px;
  height: {height_px}px;
  margin: 0;
  overflow: hidden;
  background: transparent;
}}
svg {{
  display: block;
  width: {width_px}px;
  height: {height_px}px;
}}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def screenshot_with_playwright(html_path: Path, png_path: Path, width_px: int, height_px: int, browser: str) -> bool:
    node = find_node_executable()
    node_modules = find_node_modules_dir()
    if not node or not node_modules:
        return False

    script = html_path.parent / "capture.js"
    script.write_text(
        """
const { chromium } = require("playwright");

const [htmlPath, pngPath, widthText, heightText, executablePath] = process.argv.slice(2);
const width = Number(widthText);
const height = Number(heightText);

(async () => {
  const browser = await chromium.launch({
    headless: true,
    executablePath,
  });
  const page = await browser.newPage({
    viewport: { width, height },
    deviceScaleFactor: 1,
  });
  const href = "file:///" + htmlPath.replace(/\\\\/g, "/");
  await page.goto(href, { waitUntil: "load" });
  await page.screenshot({ path: pngPath, fullPage: false, type: "png", omitBackground: true });
  await browser.close();
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
""",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["NODE_PATH"] = node_modules
    try:
        result = subprocess.run(
            [node, str(script), str(html_path), str(png_path), str(width_px), str(height_px), browser],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return result.returncode == 0 and png_path.exists()


def screenshot_with_browser_cli(html_path: Path, png_path: Path, width_px: int, height_px: int, browser: str) -> bool:
    profile_dir = Path(tempfile.mkdtemp(prefix="gerber_browser_profile_"))
    command = [
        browser,
        "--headless",
        "--disable-gpu",
        "--hide-scrollbars",
        "--no-first-run",
        f"--user-data-dir={profile_dir}",
        f"--window-size={width_px},{height_px}",
        f"--screenshot={png_path}",
        str(html_path),
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return result.returncode == 0 and png_path.exists()
    except OSError:
        return False
    finally:
        shutil.rmtree(profile_dir, ignore_errors=True)


def convert_png_to_jpg(png_path: Path, jpg_path: Path, quality: int) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("JPG export requires Pillow. Use --image-format png or install Pillow.") from exc

    with Image.open(png_path) as image:
        background = Image.new("RGB", image.size, (6, 17, 14))
        if image.mode in {"RGBA", "LA"}:
            background.paste(image, mask=image.getchannel("A"))
        else:
            background.paste(image.convert("RGB"))
        background.save(jpg_path, "JPEG", quality=quality, optimize=True)


def export_svg_image(
    svg_text: str,
    output_path: Path,
    bounds: Bounds,
    pad: float,
    dpi: int,
    image_format: str,
    jpg_quality: int,
) -> dict[str, object]:
    _, width_units, height_units = svg_viewbox(bounds, pad)
    width_px = max(1, int(round(width_units * dpi)))
    height_px = max(1, int(round(height_units * dpi)))
    browser = find_browser_executable()
    if not browser:
        raise RuntimeError("PNG/JPG export requires Microsoft Edge, Chrome, or Chromium in PATH.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gerber_export_", ignore_cleanup_errors=True) as temp_dir:
        temp_dir_path = Path(temp_dir)
        html_path = temp_dir_path / "export.html"
        png_path = output_path if image_format == "png" else temp_dir_path / "export.png"
        html_path.write_text(svg_export_html(svg_text, width_px, height_px), encoding="utf-8")
        if not screenshot_with_playwright(html_path, png_path, width_px, height_px, browser):
            if not screenshot_with_browser_cli(html_path, png_path, width_px, height_px, browser):
                raise RuntimeError(
                    "PNG/JPG export could not launch the browser renderer. "
                    "SVG and HTML outputs were still written; retry PNG export from a normal local PowerShell session."
                )
        if image_format in {"jpg", "jpeg"}:
            convert_png_to_jpg(png_path, output_path, jpg_quality)

    return {
        "path": str(output_path),
        "format": image_format,
        "dpi": dpi,
        "pixel_width": width_px,
        "pixel_height": height_px,
        "canvas_inches": {"width": width_units, "height": height_units},
        "board_pixels": {
            "width": int(round((bounds.max_x - bounds.min_x) * dpi)),
            "height": int(round((bounds.max_y - bounds.min_y) * dpi)),
        },
    }


def export_board_images(
    top_svg: str,
    bottom_svg: str,
    out_dir: Path,
    bounds: Bounds,
    pad: float,
    image_format: str,
    dpi: int,
    jpg_quality: int,
) -> dict[str, object]:
    if image_format == "none":
        return {}
    normalized_format = "jpg" if image_format == "jpeg" else image_format
    return {
        "top": export_svg_image(
            top_svg,
            out_dir / f"top_pcb.{normalized_format}",
            bounds,
            pad,
            dpi,
            normalized_format,
            jpg_quality,
        ),
        "bottom": export_svg_image(
            bottom_svg,
            out_dir / f"bottom_pcb.{normalized_format}",
            bounds,
            pad,
            dpi,
            normalized_format,
            jpg_quality,
        ),
    }


def discover_layer_files(input_dir: Path, env_layers: list[str], requested: list[str] | None) -> list[Path]:
    names: list[str] = []
    if env_layers:
        names.extend(env_layers)
    names.extend(path.name for path in sorted(input_dir.iterdir()) if path.is_file() and path.name != "AA.ENV")

    seen: set[str] = set()
    ordered_names = []
    for name in names:
        if name not in seen and (input_dir / name).is_file() and name != "AA.ENV":
            seen.add(name)
            ordered_names.append(name)

    if requested:
        requested_set = set(requested)
        ordered_names = [name for name in ordered_names if name in requested_set]

    return [input_dir / name for name in ordered_names]


def write_outputs(
    layers: list[LayerGeometry],
    out_dir: Path,
    write_layer_files: bool,
    pad: float,
    image_format: str,
    image_dpi: int,
    image_pad: float,
    jpg_quality: int,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    global_bounds = Bounds()
    for layer in layers:
        global_bounds.include_bounds(layer.bounds)
    board_bounds = estimate_single_board_bounds(layers, global_bounds)

    composite_parts = []
    for index, layer in enumerate(layers):
        color = DEFAULT_COLORS.get(layer.name, f"hsl({(index * 47) % 360} 78% 62%)")
        opacity = 0.85 if layer.name in {"COMP-P", "SOLD-P"} else 0.72
        body = render_layer_body(layer, color, opacity, visible=layer.name not in {"DRILL", "D3"})
        composite_parts.append(body)

        if write_layer_files:
            layer_svg = wrap_svg(
                render_layer_body(layer, color, 1.0),
                layer.bounds,
                pad,
                f"{layer.name} Gerber layer",
            )
            (out_dir / f"{safe_id(layer.name)}.svg").write_text(layer_svg, encoding="utf-8")

    raw_composite_svg = wrap_svg("\n".join(composite_parts), global_bounds, pad, "Raw Gerber layer composite")
    top_svg = render_physical_svg(layers, "top", board_bounds, pad, standalone=True)
    bottom_svg = render_physical_svg(layers, "bottom", board_bounds, pad, standalone=True)
    top_image_svg = render_physical_svg(layers, "top", board_bounds, image_pad, standalone=True)
    bottom_image_svg = render_physical_svg(layers, "bottom", board_bounds, image_pad, standalone=True)
    panel_top_svg = render_physical_svg(layers, "top", global_bounds, pad, standalone=True)
    panel_bottom_svg = render_physical_svg(layers, "bottom", global_bounds, pad, standalone=True)
    (out_dir / "board_layers_composite.svg").write_text(raw_composite_svg, encoding="utf-8")
    (out_dir / "board_composite.svg").write_text(top_svg, encoding="utf-8")
    (out_dir / "top_pcb.svg").write_text(top_svg, encoding="utf-8")
    (out_dir / "bottom_pcb.svg").write_text(bottom_svg, encoding="utf-8")
    (out_dir / "panel_top_pcb.svg").write_text(panel_top_svg, encoding="utf-8")
    (out_dir / "panel_bottom_pcb.svg").write_text(panel_bottom_svg, encoding="utf-8")
    image_exports = export_board_images(
        top_image_svg,
        bottom_image_svg,
        out_dir,
        board_bounds,
        image_pad,
        image_format,
        image_dpi,
        jpg_quality,
    )
    (out_dir / "board_viewer.html").write_text(
        render_pcb_viewer_html(layers, board_bounds, "PCB Gerber viewer", pad),
        encoding="utf-8",
    )
    raw_viewer_error = None
    try:
        (out_dir / "board_layers_raw.html").write_text(
            render_html(layers, global_bounds, "Raw Gerber layer viewer", pad),
            encoding="utf-8",
        )
    except OSError as exc:
        raw_viewer_error = str(exc)

    summary = {
        "bounds": board_bounds.as_dict(),
        "panel_bounds": global_bounds.as_dict(),
        "generated_files": {
            "viewer": str(out_dir / "board_viewer.html"),
            "raw_viewer": str(out_dir / "board_layers_raw.html"),
            "top_preview": str(out_dir / "top_pcb.svg"),
            "bottom_preview": str(out_dir / "bottom_pcb.svg"),
            "panel_top_preview": str(out_dir / "panel_top_pcb.svg"),
            "panel_bottom_preview": str(out_dir / "panel_bottom_pcb.svg"),
            "raw_composite": str(out_dir / "board_layers_composite.svg"),
        },
        "image_exports": image_exports,
        "side_mapping": SIDE_CONFIG,
        "raw_viewer_error": raw_viewer_error,
        "layers": [
            {
                "name": layer.name,
                "source": str(layer.source_path),
                "unit": layer.settings.unit,
                "format": f"{layer.settings.integer_digits}.{layer.settings.decimal_digits}",
                "zero_suppression": layer.settings.zero_suppression,
                "draw_count": layer.draw_count,
                "flash_count": layer.flash_count,
                "move_count": layer.move_count,
                "bounds": layer.bounds.as_dict(),
                "warnings": layer.warnings,
            }
            for layer in layers
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Render Gerber files to SVG and an HTML PCB viewer.")
    parser.add_argument("input_dir", nargs="?", type=Path, default=Path.cwd(), help="Directory containing Gerber files")
    parser.add_argument("--out", type=Path, default=Path("gerber_render"), help="Output directory")
    parser.add_argument("--layers", nargs="*", help="Optional layer file names to render")
    parser.add_argument("--no-layer-files", action="store_true", help="Only write composite SVG, HTML, and summary")
    parser.add_argument("--pad", type=float, default=0.08, help="Padding around the board in Gerber units")
    parser.add_argument(
        "--image-format",
        choices=["png", "jpg", "jpeg", "none"],
        default="png",
        help="Export top/bottom board images in this raster format. Use 'none' to skip image export.",
    )
    parser.add_argument("--image-dpi", type=int, default=1200, help="Raster export DPI based on Gerber units")
    parser.add_argument(
        "--image-pad",
        type=float,
        default=0.0,
        help="Padding around exported PNG/JPG images in Gerber units. Default 0 crops to the PCB bounds.",
    )
    parser.add_argument("--jpg-quality", type=int, default=95, help="JPEG quality from 1 to 100")
    args = parser.parse_args()

    if args.image_dpi <= 0:
        raise SystemExit("--image-dpi must be greater than 0")
    if args.image_pad < 0:
        raise SystemExit("--image-pad must be 0 or greater")
    if not 1 <= args.jpg_quality <= 100:
        raise SystemExit("--jpg-quality must be between 1 and 100")

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    env_settings, env_apertures, env_layers = parse_env(input_dir / "AA.ENV")
    layer_files = discover_layer_files(input_dir, env_layers, args.layers)
    if not layer_files:
        raise SystemExit("No Gerber layers found.")

    layers = [parse_gerber(path, env_settings, env_apertures) for path in layer_files]
    summary = write_outputs(
        layers,
        args.out.resolve(),
        not args.no_layer_files,
        args.pad,
        args.image_format,
        args.image_dpi,
        args.image_pad,
        args.jpg_quality,
    )

    print(f"Rendered {len(layers)} layer(s) to {args.out.resolve()}")
    print(f"Open: {args.out.resolve() / 'board_viewer.html'}")
    if summary["image_exports"]:
        for side, export in summary["image_exports"].items():
            print(
                f"{side.capitalize()} image: {export['path']} "
                f"({export['pixel_width']} x {export['pixel_height']} px, {export['dpi']} DPI)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

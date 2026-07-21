"""Build a Leave-One-Board-Out fold's REAL dataset for SD-DETR.

Re-splits CustomPCB_full_class4 by board: the --test-board goes to val, all
other boards go to train. Gerber templates (referenced by template_file_name,
not COCO image entries) are copied into both splits so the twin network can
load them. COCO train/val jsons are rebuilt with contiguous ids.

Output layout (what main.py --custom_pcb_path expects):
  <out>/images/{train,val}/*.jpg        (board images)
  <out>/images/{train,val}/gerber_*.png (templates, both splits)
  <out>/annotations/custom_{train,val}_class4.json
"""
import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_fold(source, test_board, out):
    source = Path(source)
    out = Path(out)

    images_by_fn = {}
    anns_by_fn = defaultdict(list)
    categories = None
    for split in ("train", "val"):
        d = load(source / "annotations" / f"custom_{split}_class4.json")
        categories = d["categories"]
        id2fn = {im["id"]: im["file_name"] for im in d["images"]}
        for im in d["images"]:
            images_by_fn[im["file_name"]] = im
        for a in d["annotations"]:
            anns_by_fn[id2fn[a["image_id"]]].append(a)

    val_fns = sorted(fn for fn in images_by_fn if fn.split("_")[0] == test_board)
    train_fns = sorted(fn for fn in images_by_fn if fn.split("_")[0] != test_board)
    if not val_fns:
        raise SystemExit(f"No images found for test board '{test_board}'.")

    def find_src(file_name):
        for split in ("train", "val"):
            p = source / "images" / split / file_name
            if p.exists():
                return p
        raise FileNotFoundError(f"Source image not found on disk: {file_name}")

    def make_coco(fns):
        images, annotations = [], []
        image_id, ann_id = 1, 1
        for fn in fns:
            im = dict(images_by_fn[fn])
            im["id"] = image_id
            images.append(im)
            for a in anns_by_fn.get(fn, []):
                a2 = dict(a)
                a2["id"] = ann_id
                a2["image_id"] = image_id
                if "group_id" in a2:
                    a2["group_id"] = image_id
                annotations.append(a2)
                ann_id += 1
            image_id += 1
        return {"images": images, "annotations": annotations, "categories": categories}

    # gerber templates present in either source split (deduped by name)
    gerbers = {}
    for split in ("train", "val"):
        for g in (source / "images" / split).glob("gerber_*.png"):
            gerbers[g.name] = g

    if out.exists():
        shutil.rmtree(out)
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    for split, fns in (("train", train_fns), ("val", val_fns)):
        img_dir = out / "images" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        for fn in fns:
            shutil.copy2(find_src(fn), img_dir / fn)
        for name, g in gerbers.items():
            shutil.copy2(g, img_dir / name)
        coco = make_coco(fns)
        (out / "annotations" / f"custom_{split}_class4.json").write_text(
            json.dumps(coco, indent=2), encoding="ascii"
        )
        n_ann = sum(len(anns_by_fn.get(fn, [])) for fn in fns)
        print(f"[{split}] board-images={len(fns)} gerbers={len(gerbers)} defect-anns={n_ann}")

    print(f"Fold real dataset (test board={test_board}) -> {out}")


def main():
    ap = argparse.ArgumentParser(description="Build one LOBO fold's real dataset.")
    ap.add_argument("--source", default="/home/jhj/data/CustomPCB_full_class4")
    ap.add_argument("--test-board", required=True, help="e.g. num1")
    ap.add_argument("--out", required=True, help="e.g. data/CustomPCB_lobo/fold_num1/real")
    a = ap.parse_args()
    build_fold(a.source, a.test_board, a.out)


if __name__ == "__main__":
    main()

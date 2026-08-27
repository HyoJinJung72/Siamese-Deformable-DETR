# Experiments

Complete record of experiments backing the paper. **Every number claimed in the paper must come
from this file.** Large artifacts (checkpoints, feature maps) are not tracked in git; only the
resulting numbers are recorded here.

---

## 0. Setup

| Item | Value |
|---|---|
| Base detector | Deformable DETR (ResNet-50), 4 feature levels |
| Init | COCO-pretrained Deformable DETR (`r50_deformable_detr-checkpoint.pth`) |
| Optimizer / LR | AdamW, `lr 2e-4`, `lr_backbone 2e-5`, weight decay `1e-4` |
| Schedule | 50 epochs, LR drop at 40, batch size 2 |
| Losses | cls 2, L1 bbox 5, GIoU 2 |
| TCDF init | fusion convs zero-initialized; gate init 0.05 |
| Hardware | single NVIDIA RTX A5000 |
| Seeds | 5 seeds (0–4) where mean±std is reported |

Model variants: `single` (no flags) · `basic` (`--use_template`) · `TCDF` (`--use_template --use_tcdf`).

---

## 1. Datasets

### DeepPCB (benchmark)
Paired defect-free template / defective test images, 6 defect classes.

| Split | Images | Annotations |
|---|---|---|
| Train | 1,000 | 6,873 |
| Test  | 500 | 3,140 |

Per class (train / test): open 1283/659 · short 1028/478 · mousebite 1379/586 ·
spur 1142/483 · copper 1010/464 · pin-hole 1031/470.

### CustomPCB (collected from a real production line)
One Gerber design; 8 defective boards + 1 normal board; both sides split into 32 tiles
→ **576 images**, 4 classes. Only **147 images contain defects**.

| Class | Description | Train / Test annotations |
|---|---|---|
| Silk | silkscreen encroaching on pad or trace | 23 / 6 |
| Short | unintended conductive bridge | 50 / 6 |
| Pad_open | insufficient solder on pad surface | 20 / 5 |
| Open | broken trace | 40 / 5 |
| **Total** | | **133 / 22** |

Main split: **PCB 3 as test**, remaining boards as train. Leave-one-board-out folds are built with
`build_lobo_fold.py`.

### Gerber template rendering
A custom Gerber→image converter (`tools/gerber_visualizer.py`) is used because KiCAD / GerbView
renderings do not preserve fine circuit detail. The custom renderer produces realistic PCB-like
images, which also reduces the appearance gap to the test images at the *input* level.

**Converter-comparison datasets (built, experiments pending).** To later compare template
converters under identical test images/annotations, two drop-in replacements of the num3 fold were
built where only the Gerber template images change: `CustomPCB_kicad` and `CustomPCB_gerbview`
(the 4 full-board KiCAD/GerbView renders were sliced into the same 8×4 grid; the KiCAD bottom side
is horizontally mirrored to match, determined by edge correlation). Test images and annotations are
byte-identical to `CustomPCB_lobo/fold_num3`. **No training/eval has been run on these yet**, so no
results are recorded. Caveat: the supplied full-board renders are low-resolution (~729×350 /
854×410), so the sliced templates are ~5× upsampled — any future comparison confounds converter
appearance with resolution.

---

## 2. Main result — DeepPCB

single / basic / TCDF / TCDF+CB, 5 seeds each (mean over seeds).

| Model | AP50 | AP75 | AP50-95 | Template |
|---|---|---|---|---|
| Ghost-YOLOv8 | 92.3 | – | – | ✗ |
| YOLO-MSCA | 97.5 | – | 75.5 | ✗ |
| YOLO-HB | 98.4 | – | 72.4 | ✗ |
| YOLOv8-DEE | 98.7 | 91.6 | 72.9 | ✗ |
| FMW-YOLO | 98.9 | – | 78.9 | ✗ |
| YOLO-HDEW | 98.9 | – | 80.1 | ✗ |
| LPViT | 98.8 | **97.6** | – | ✗ |
| SF-PSPyramid | 98.6 | 94.6 | – | ✗ |
| DeepPCB | 98.6 | – | – | ✓ |
| PIDDN | 98.4 | 84.3 | 69.8 | ✓ |
| **Ours (TCDF)** | **99.0** | 95.2 | **81.5** | ✓ |

**Claimable**: best **AP50** and best **AP50-95** among both template-using and template-free
methods; AP50-95 is **+1.4 %p** over the best prior (80.1).
**Not claimable**: best AP75 (LPViT 97.6 is higher).

Qualitative: the single-network baseline misses defects that TCDF detects. Quantified by matching
predictions to GT (IoU ≥ 0.5, score ≥ 0.3) across the 500 test images:
**47 images contain at least one defect missed by single but caught by TCDF; 43 of those have no
reverse case** (nothing lost). Rescued defects are dominated by **short** and **spur** (thin, small).
Clearest samples: `90100033`, `44000052`, `90100052`, `90100051`, `90100066`, `90100029`.

---

## 3. Main result — CustomPCB (test = PCB 3)

AP50 overall and per class.

| Method | Template | AP50 | Silk | Short | Pad_open | Open |
|---|---|---|---|---|---|---|
| Single network | ✗ | 0.258 | 0.312 | 0.031 | 0.126 | 0.562 |
| Basic template fusion | ✓ | 0.430 | 0.737 | 0.227 | 0.280 | 0.477 |
| **Ours (TCDF)** | ✓ | **0.527** | 0.686 | 0.162 | 0.604 | 0.654 |
| Single + Augmentation | ✗ | 0.479 | 0.569 | 0.144 | 0.741 | 0.460 |
| Basic + Augmentation | ✓ | 0.538 | 0.655 | 0.103 | 0.788 | 0.607 |
| **Ours + Augmentation** | ✓ | **0.609** | 0.804 | 0.203 | 0.728 | 0.700 |

Reading: Single → Basic isolates the value of *using a reference*; Basic → TCDF isolates the value
of *our fusion design*. CutPaste augmentation adds a further gain on top.

---

## 4. Template ablation (evaluated with the TCDF model, CustomPCB)

Same trained checkpoint, only the template input is perturbed at evaluation
(`--template_ablation_mode`).

| Template condition | Description | AP50 | ΔAP50 |
|---|---|---|---|
| Correct (paired gerber) | paired gerber reference | 0.527 | – |
| Zero | constant (zero) input instead of gerber | 0.096 | −0.431 |
| Test as template | test image itself as reference | 0.001 | −0.526 |
| Shuffled (mismatched) | gerber of a *different* image (deterministic index offset) | 0.063 | −0.464 |

**Interpretation.**
1. All three perturbations collapse → the gain requires the *correct* reference, not merely an
   extra branch or more data.
2. **Shuffled (0.063) < Zero (0.096)** → a wrong reference is *worse than no information*.
3. **Test-as-template collapses to 0.001** → with the reference equal to the test image the
   difference signal vanishes, showing the model genuinely depends on the reference–test difference.

> "Shuffled" is **not** a random shuffle: each image is paired with the template of the image at a
> fixed index offset (offset 1, with a guard so an image never gets its own template). So it is a
> consistently *mispaired* reference — a different slice is always used, never the correct one.

---

## 5. Robustness to template–inspection misalignment (DeepPCB)

Reference-based inspection assumes the template and test image are aligned; in practice small
translation/rotation offsets occur. We measure robustness on **DeepPCB** (3,140 test annotations
give stable measurements). Perturbation ranges are motivated by reported PCB registration
errors (well-registered residual ≈ sub-pixel to ~1 px; unfixtured/poorly-registered up to several–
tens of px).

**Two mechanisms (no architecture change, no extra parameters, no inference cost):**
- *Train-time injection* — at each iteration a random translation+rotation is applied to the
  **template only** (`--train_template_misalign_translate/rotate`); the test image and boxes are
  untouched. **Weak** = up to ±8 px / ±3°, **Strong** = up to ±16 px / ±5°.
- *Eval-time sweep* — the **template only** is shifted by a fixed amount at evaluation
  (`--eval_template_perturb`); test image and annotations are unchanged, so COCO scoring stays valid
  and a reference-free model is invariant by construction (a sanity control).

**AP50 under a template-only perturbation sweep** (seed 0):

| Method | t0 | t2 | t4 | t8 | t16 | | r0 | r1 | r2 | r3 | r4 | r5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Without injection | 99.0 | 98.8 | 98.3 | 92.3 | 74.2 | | 99.0 | 98.6 | 96.2 | 91.0 | 86.0 | 80.9 |
| Weak (±8px/±3°) | 98.4 | 98.5 | 98.6 | 98.4 | **97.4** | | 98.4 | 98.6 | 98.6 | 98.5 | 98.2 | **97.9** |
| Strong (±16px/±5°) | 98.2 | 98.2 | 98.1 | 98.0 | 97.6 | | 98.2 | 98.3 | 98.4 | 98.3 | 98.3 | 98.1 |

*Translation in px (t); rotation in degrees (r); r0 = t0 = no perturbation. AP50-95 follows the same
trend (e.g. without-injection 81.5→55.6 at t16; injected stays ≈78–80).*

**Findings.**
1. Without injection, performance collapses as misalignment grows (99.0→74.2 at 16 px, →80.9 at 5°).
2. Injection keeps AP50 at **≥97.4** under the same shifts, at a small aligned-case cost (99.0→98.4).
3. **Weak ≈ Strong** (97.4 vs 97.6 at 16 px; 97.9 vs 98.1 at 5°) while Strong costs slightly more at
   t0 (98.2 vs 98.4) → a **small injection range suffices**; the weak setting is preferred.
4. Robustness generalizes slightly beyond the trained range (weak trained to ±3° still holds at 5°).

**Framing for the paper.** This is a tolerance/robustness result, not registration/correction — the
template stays misaligned; the model just becomes insensitive within (and near) the trained range.

### 5.1 CustomPCB misalignment — measured but unreliable (excluded from claims)

The same protocol was run on CustomPCB (baseline / weak ±2px·±3° / strong ±4px·±5°, all from the
DeepPCB-TCDF checkpoint for a fair base). **Results are noise-dominated and non-monotonic**: the
without-injection baseline barely degrades (0.402→0.351 over 0–4 px) and the rotation sweep is
non-monotonic (e.g. r3 > r0), because the 22-annotation validation set moves AP by ~4.5 %p per
detection — larger than the misalignment signal. The "baseline degrades → injection recovers"
contrast therefore does not appear. **Use DeepPCB only for this experiment**; the justification is
measurement precision (this experiment measures a fine, continuous degradation curve, unlike the
large discrete effects in Sections 3–4 that CustomPCB can resolve), not a defect of the dataset.

---

## 6. Computational cost

`tools/bench_cost.py` — measured on **CustomPCB** (input 3×384×401, batch 1, 300 iters, 20 warm-up
discarded). Template-using variants forward the template through the backbone as well (cost fairly
included). **#Params is dataset-independent; FLOPs / Latency / FPS scale with input resolution**, so
on DeepPCB (640×640) FLOPs/latency would be higher and FPS lower — these numbers are CustomPCB-specific.

| Method | #Params | FLOPs | Latency | FPS |
|---|---|---|---|---|
| Single network | 40.05 M | 63.9 G | 21.5 ms | 46.5 |
| Basic template fusion | 40.57 M | 92.0 G | 28.9 ms | 34.6 |
| Ours (TCDF) | 41.36 M | 93.3 G | 29.4 ms | 34.0 |

**Key asymmetry**: parameters grow only **+3.3 %** (backbone weights are shared), while FLOPs/latency
grow ~46 % / ~37 % — this increase is the *intrinsic cost of passing the reference through the
backbone once more*, not new heavy modules. Still **34 FPS**, above the 30 FPS real-time bar.
TCDF over Basic costs only +0.8 M params and +0.5 ms.

> FLOPs exclude the custom deformable-attention CUDA kernel (identical across all three variants,
> so relative comparison is valid).

---

## 7. Results by test PCB (CustomPCB)

Main experiments use PCB 3 as the test set; this table simply reports what happens when a different
board is held out. **This is a report, not a generalization claim** — scores vary with which classes
appear in each board.

| Test | Classes present | AP50 | Silk | Short | Pad_open | Open |
|---|---|---|---|---|---|---|
| PCB1 | Open | 0.343 | – | – | – | 0.343 |
| PCB2 | Silk, Short, Pad_open, Open | 0.293 | 0.723 | 0.003 | 0.109 | 0.337 |
| PCB4 | Pad_open | 0.627 | – | – | 0.627 | – |
| PCB5 | Short | 0.159 | – | 0.159 | – | – |
| PCB6 | Silk | 0.443 | 0.443 | – | – | – |
| PCB7 | Silk, Short, Pad_open, Open | 0.508 | 0.508 | 0.322 | 0.663 | 0.505 |
| PCB8 | Short, Open | 0.355 | – | 0.118 | – | 0.593 |

---

## 8. Internal analysis of TCDF

### 8.1 Learned gate values (`σ(α_l)`)

| Dataset | level 0 | level 1 | level 2 | level 3 |
|---|---|---|---|---|
| CustomPCB | 0.056 | 0.051 | 0.050 | 0.052 |
| DeepPCB | 0.056 | 0.051 | 0.050 | 0.052 |

The gate stays near its initialization (0.05), i.e. the fused correction is ≈95 % context branch.
Difference-branch conv weights are nevertheless large (L2 norm: CustomPCB 14.7 / 12.0 / 9.4 / 8.2;
DeepPCB 12.4 / 9.0 / 6.7 / 5.8), so the branch produces a strong internal signal that is then
scaled down. **Caveat**: because the branch conv can absorb scale, `g` alone does not prove the
branch's true contribution (identifiability).

### 8.2 Feature-map concentration on defects

Mean ratio of activation inside GT boxes to outside (`>1` = concentrated on defects),
from `--save_eval_tffn_analysis`.

| Map | CustomPCB | DeepPCB |
|---|---|---|
| template `F^R` | 1.07 | 1.10 |
| test `F^T` | 1.07 | 1.43 |
| **\|F^R − F^T\|** | **1.28** | **6.88** |
| Δ_context | 1.07 | 1.32 |
| **Δ_diff** | **1.27** | **3.82** |
| Δ (combined) | 1.08 | 1.38 |
| F^fused | 1.08 | 1.43 |

**Interpretation.** The role separation is real and measurable: the **difference branch concentrates
on defects** (1.27 / 3.82) while the **context branch is nearly uniform** (1.07 / 1.32). The raw
`|F^R − F^T|` is the most defect-concentrated signal, which justifies feeding it explicitly.
DeepPCB shows a much sharper separation than CustomPCB (binary images vs real photographs).

Best CustomPCB samples for the figure (ranked by difference-branch concentration and role
separation): `num3_defect_front_04` (clearest separation: Δ_context 0.99 vs Δ_diff 1.43),
then `num3_defect_back_29` (1.16 / 1.67), `back_28`, `back_13`, `front_12`, `back_08`.

---

## 9. Synthetic defect generation (CutPaste-based)

**Motivation.** Only 147 of 576 CustomPCB images contain defects; defect samples are rare and
expensive in real production, which bounds model performance.

**Method.** A real defect region is cut from a defective board and pasted onto a normal board.
Unlike the original CutPaste (random placement):
- **Registered placement** — the target board is aligned to the reference by phase correlation
  (translation offset), so the defect lands on the corresponding circuit location.
- **Feather blending** — the defect mask edge is Gaussian-softened (σ = 1.0) so no hard paste seam
  remains and the model cannot learn the paste artifact instead of the defect.

**Counts (num3 fold, `CustomPCB_cutpaste_num3`).** Target was 100 synthetic samples per class;
actual = **Open 100 / Short 100 / Silk 100 / Pad_open 96** (total **396**). Pad_open stops at 96
because `--no-combination-recycle` halts once the unique (normal × reference-defect × slice)
combinations are exhausted (recycling would only emit pixel-identical duplicates). Synthetics are
appended to **train only**; val stays real-only (num3). Standalone, byte-for-byte reproducible via
`tools/cutpaste_custom_pcb.py` (verified: regenerates the dataset identically for a fixed seed).

Effect: see the "+ Augmentation" rows in Section 3 (Ours 0.527 → 0.609).

---

## 10. Reproduction

```bash
# DeepPCB — TCDF (repeat over seeds 0-4 for mean±std)
python main.py --dataset_file deeppcb --deep_pcb_path /path/to/DeepPCB \
  --pretrained pretrained/r50_deformable_detr-checkpoint.pth \
  --use_template --use_tcdf --epochs 50 --lr_drop 40 --batch_size 2 --seed 0 \
  --output_dir output/deeppcb_tcdf

# CustomPCB — TCDF
python main.py --dataset_file custompcb_class4_full --custom_pcb_path /path/to/CustomPCB \
  --pretrained pretrained/r50_deformable_detr-checkpoint.pth \
  --use_template --use_tcdf --epochs 50 --lr_drop 40 --batch_size 2 \
  --output_dir output/custompcb_tcdf

# Template ablation (repeat with zero / test_as_template / shuffled)
python main.py --dataset_file custompcb_class4_full --custom_pcb_path /path/to/CustomPCB \
  --use_template --use_tcdf --template_ablation_mode normal \
  --resume output/custompcb_tcdf/checkpoint_best.pth --eval --output_dir output/tmpl_correct

# Misalignment robustness — train with template injection, then sweep at eval (DeepPCB)
python main.py --dataset_file deeppcb --deep_pcb_path /path/to/DeepPCB \
  --pretrained pretrained/r50_deformable_detr-checkpoint.pth --use_template --use_tcdf \
  --train_template_misalign_translate 8 --train_template_misalign_rotate 3 \
  --epochs 50 --lr_drop 40 --batch_size 2 --output_dir output/deeppcb_tcdf_misalign
python main.py --dataset_file deeppcb --deep_pcb_path /path/to/DeepPCB --use_template --use_tcdf \
  --resume output/deeppcb_tcdf_misalign/checkpoint_best.pth --eval \
  --eval_template_perturb translate --eval_template_perturb_dx 8 --eval_template_perturb_dy 8 \
  --output_dir output/deeppcb_tcdf_misalign_t8
# sweep scripts: tools/run_phase1_alignment.sh, tools/eval_align_model.sh,
#                tools/collect_phase1_alignment.py

# Cost benchmark (run from repo root; measured on CustomPCB)
python tools/bench_cost.py --dataset_file custompcb_class4_full --custom_pcb_path /path/to/CustomPCB \
  --use_template --use_tcdf

# Feature-map analysis (TCDF internals)
python main.py ... --use_template --use_tcdf --save_eval_tffn_analysis --eval_tffn_level 0 \
  --resume <ckpt> --eval --output_dir output/tcdf_analysis

# Custom Gerber rendering / CutPaste synthesis
python tools/gerber_visualizer.py /path/to/gerber_dir --out output/gerber_render
python tools/cutpaste_custom_pcb.py --data-root /path/to/CustomPCB_class4 \
  --mask-xml /path/to/annotations_class4.xml --output-dir output/CustomPCB_cutpaste ...

# Leave-one-board-out fold
python tools/build_lobo_fold.py --test-board num3 --out /path/to/CustomPCB_lobo/fold_num3
```

---

## 11. Explored and rejected

Keep these out of the paper unless new evidence appears; they were implemented and tested.

| Approach | Outcome |
|---|---|
| TFRM (attention-based template rectification, from PIDDN) | no measurable gain in our setting |
| Feature suppression module | no gain |
| Image-transform module (Gerber→PCB domain) | no gain |
| Reliability / consistency sample weighting | no gain |
| Box refinement (`--with_box_refine`) on DeepPCB | ineffective; AP75 already near ceiling |
| Raising the GIoU loss coefficient (2 → 4) | no effect; localization is limited by feature resolution, not loss weighting |
| Earlier TF-IDG augmentation (v1) | hurt difference-based methods — synthetic defects had an unrealistic difference signature |

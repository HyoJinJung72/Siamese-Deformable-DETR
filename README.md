# Reference-Based PCB Defect Detection via Siamese Deformable DETR with Template Images

Reference-based PCB defect detection that compares an inspection image against the
board's **Gerber design (template)** to localize defects as *deviations from the intended design*.
The model couples a **weight-sharing Siamese backbone** with a **Template Context–Difference
Fusion (TCDF)** module and a **Deformable DETR** detector.

## Highlights

- **Gerber as a defect-free reference.** The Gerber file is the board's design drawing, so the
  reference is defect-free by construction (no need for hand-screened golden samples).
- **TCDF fusion.** Instead of pixel subtraction or plain feature concatenation, TCDF uses the
  **difference between reference and inspection features** as an explicit defect signal:
  - *Context branch* — `Conv1x1([F_R; F_T])` encodes the normal circuit structure.
  - *Difference branch* — `Conv1x1([F_R; F_T; |F_R - F_T|])` highlights where the board deviates.
  - *Learnable gate* — a per-scale gate `g` adaptively fuses the two branches, added as a
    residual to the inspection feature (`F_fused = F_T + Delta`).
- **Feature-space comparison** absorbs template/inspection appearance gaps (pattern width, color),
  so normal regions yield near-zero difference and only real defects are emphasized.

## Method

```
Template (Gerber) --+
                    +-- Shared Backbone -- multi-scale features -- TCDF -- Deformable DETR -- class + box
Inspection (Test) --+        (Siamese)      F_R, F_T (x4)      (context+difference+gate)
```

## Installation

Built on [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).

```bash
# 1) environment (PyTorch with CUDA)
pip install -r requirements.txt

# 2) compile the MultiScaleDeformableAttention CUDA op
cd models/ops
sh make.sh
python test.py   # optional: unit test
cd ../..
```

## Data preparation

Datasets are **not** included in this repository. Expected layouts:

**DeepPCB** - `--dataset_file deeppcb --deep_pcb_path /path/to/DeepPCB`
(paired defect-free template / defective test images with COCO-style annotations).

**CustomPCB** - `--dataset_file custompcb_class4_full --custom_pcb_path /path/to/CustomPCB`

The CustomPCB dataset is available for download here:
[Google Drive]([https://drive.google.com/file/d/170-tRWLfrEiDTXvxuHT7AuTwjCv7BDOm/view?usp=drive_link](https://drive.google.com/file/d/12y8uy-cCPe-vEGSthVuFYk9Y5ony5pdi/view?usp=drive_link)).

```
CustomPCB/
|-- images/{train,val}/            # inspection images + gerber templates
`-- annotations/
    |-- custom_train_class4.json   # 4 classes: open, pad_open, silk, short
    `-- custom_val_class4.json
```
Each image entry carries a `group_image_list` pointing to its Gerber template.
Leave-one-board-out folds can be generated with `tools/build_lobo_fold.py`.

### Gerber file converter

`tools/gerber_visualizer.py` renders a directory of RS-274D/RS-274X Gerber files into a
PCB-like image that is used as the defect-free reference (template). It parses the design
layers (copper, solder mask, silkscreen, drill) and composites them with realistic colors and
shading so that the rendered template resembles the captured inspection image, reducing the
template/inspection appearance gap at the input level.

```bash
# Render a Gerber directory to images
python tools/gerber_visualizer.py /path/to/gerber_dir --out output/gerber_render

# higher-resolution raster export (default 1200 DPI)
python tools/gerber_visualizer.py /path/to/gerber_dir --out output/gerber_render --image-dpi 1200
```

The input directory should contain the Gerber layer files
(e.g., `COMP-P`, `SOLD-P`, `C-MASK`, `S-MASK`, `C-SILK`, `S-SILK`, `DRILL`) together with the
`AA.ENV` aperture/metadata file. Outputs include the composite SVG, an HTML PCB viewer, and a
rasterized PNG/JPG that is used as the template image.

## Usage

Three model variants share the same command; flags select the fusion:

| Variant | Flags |
|---|---|
| Single network (no reference) | *(none)* |
| Basic template fusion | `--use_template` |
| **TCDF (ours)** | `--use_template --use_tcdf` |

### DeepPCB

```bash
# Train (TCDF)
python main.py --dataset_file deeppcb --deep_pcb_path /path/to/DeepPCB \
  --pretrained pretrained/r50_deformable_detr-checkpoint.pth \
  --use_template --use_tcdf \
  --epochs 50 --lr_drop 40 --batch_size 2 --output_dir output/deeppcb_tcdf

# Evaluate
python main.py --dataset_file deeppcb --deep_pcb_path /path/to/DeepPCB \
  --use_template --use_tcdf --resume output/deeppcb_tcdf/checkpoint_best.pth --eval \
  --output_dir output/deeppcb_tcdf_eval
```

### CustomPCB

```bash
python main.py --dataset_file custompcb_class4_full --custom_pcb_path /path/to/CustomPCB \
  --pretrained pretrained/r50_deformable_detr-checkpoint.pth \
  --use_template --use_tcdf \
  --epochs 50 --lr_drop 40 --batch_size 2 --output_dir output/custompcb_tcdf
```

### Misalignment robustness

Reference-based inspection assumes the reference and test images are aligned; in practice small
translation/rotation offsets occur. Two independent mechanisms are provided:

- **Train time** — inject a small random offset into the *template only* so the model learns to
  tolerate misregistration (no architecture change, no extra parameters):
  `--train_template_misalign_translate <px> --train_template_misalign_rotate <deg>`
- **Eval time** — perturb the *template only* (test image and boxes untouched, so COCO scoring
  stays valid and a reference-free model is invariant by construction):
  `--eval_template_perturb {translate,rotate,scale,affine} --eval_template_perturb_dx/dy/angle <v>`

```bash
# Train with misalignment injection (translate +/-8px, rotate +/-3deg)
python main.py --dataset_file deeppcb --deep_pcb_path /path/to/DeepPCB \
  --use_template --use_tcdf --train_template_misalign_translate 8 --train_template_misalign_rotate 3 \
  --epochs 50 --lr_drop 40 --batch_size 2 --output_dir output/deeppcb_tcdf_misalign

# Evaluate a template-shift sweep (delta = 8px)
python main.py --dataset_file deeppcb --deep_pcb_path /path/to/DeepPCB \
  --use_template --use_tcdf --resume output/deeppcb_tcdf_misalign/checkpoint_best.pth --eval \
  --eval_template_perturb translate --eval_template_perturb_dx 8 --eval_template_perturb_dy 8 \
  --output_dir output/deeppcb_tcdf_misalign_d8
```

`tools/run_phase1_alignment.sh` and `tools/eval_align_model.sh` run the full translation/rotation
sweeps; `tools/collect_phase1_alignment.py` tabulates the AP results.

### Synthetic defect generation (CutPaste)

`tools/cutpaste_custom_pcb.py` builds the synthetic-defect training data used on CustomPCB. A real
defect is cut from a defective board and pasted onto a normal board with **registered placement**
(phase-correlation alignment to the matching circuit location) and **feather blending**
(`--defect-blend-sigma`, a Gaussian-softened paste boundary so the model learns the defect, not the
seam). It needs the raw CustomPCB boards + a CVAT mask XML, and reproduces the dataset
deterministically for a given `--seed`.

```bash
python tools/cutpaste_custom_pcb.py \
  --data-root /path/to/CustomPCB_class4 --mask-xml /path/to/annotations_class4.xml \
  --output-dir /path/to/out/CustomPCB_cutpaste --splits train val \
  --defect-boards num1,...,num8 --normal-boards num1,...,num9 --exclude-boards num3 \
  --normal-source all_normal --placement aligned --require-aligned-target \
  --ref-policy same_class --ref-match-scope same_sample \
  --combination-sampling unique_cycle --no-combination-recycle \
  --composite-mode cutpaste --cutpaste-placement registered \
  --source-blend feather --defect-blend-sigma 1.0 --samples-per-class 100 --seed 20260703
```

### Useful options

- `--template_ablation_mode {normal,zero,test_as_template,shuffled}` - template ablation (checks
  whether the *correct* Gerber pairing is what drives the gains).
- `--class_balanced_sampling sqrt_inverse` - image-level balanced sampling for minority classes.
- `--save_eval_cam` / `--save_eval_tffn_analysis` - save CAM / TCDF feature-map visualizations.

On CustomPCB (AP50): Single 0.258 -> Basic 0.430 -> **TCDF 0.527**, and **0.609** with CutPaste augmentation.

## Repository layout

```
main.py                     # train / eval entry
engine.py                   # train_one_epoch / evaluate (+ CAM / TCDF-analysis saving)
models/deformable_detr.py   # Siamese backbone + TCDF + detector (build_model)
datasets/                   # deeppcb.py, custompcb*.py, transforms, coco utils
tools/build_lobo_fold.py    # leave-one-board-out fold builder (CustomPCB)
tools/bench_cost.py         # #Params / FLOPs / Latency / FPS benchmark
tools/gerber_visualizer.py  # Gerber -> template image renderer
tools/cutpaste_custom_pcb.py # CutPaste synthetic defect generator
```

> Large artifacts (`output/`, `pretrained/`, checkpoints) are gitignored and not tracked.

## Acknowledgements

This code is built upon [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR)
(Zhu et al., ICLR 2021). See `LICENSE` (Apache-2.0).

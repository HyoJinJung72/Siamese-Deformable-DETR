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
```
CustomPCB/
|-- images/{train,val}/            # inspection images + gerber templates
`-- annotations/
    |-- custom_train_class4.json   # 4 classes: open, pad_open, silk, short
    `-- custom_val_class4.json
```
Each image entry carries a `group_image_list` pointing to its Gerber template.
Leave-one-board-out folds can be generated with `build_lobo_fold.py` / `build_ablation_folds.sh`.

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

`run_deeppcb_avg.sh` runs single / basic / TCDF (+ class-balanced) over 5 seeds and reports mean+/-std.

### CustomPCB

```bash
python main.py --dataset_file custompcb_class4_full --custom_pcb_path /path/to/CustomPCB \
  --pretrained pretrained/r50_deformable_detr-checkpoint.pth \
  --use_template --use_tcdf \
  --epochs 50 --lr_drop 40 --batch_size 2 --output_dir output/custompcb_tcdf
```

### Useful options

- `--template_ablation_mode {normal,zero,test_as_template,shuffled}` - template ablation (checks
  whether the *correct* Gerber pairing is what drives the gains).
- `--class_balanced_sampling sqrt_inverse` - image-level balanced sampling for minority classes.
- `--save_eval_cam` / `--save_eval_tffn_analysis` - save CAM / TCDF feature-map visualizations.

## Results (DeepPCB)

| Model | AP50 | AP75 | AP50-95 | Template |
|---|---|---|---|---|
| YOLO-HDEW | 98.9 | - | 80.1 | No |
| LPViT | 98.8 | 97.6 | - | No |
| PIDDN | 98.4 | 84.3 | 69.8 | Yes |
| **Ours (TCDF)** | **99.0** | 95.2 | **81.5** | Yes |

On CustomPCB (AP50): Single 0.258 -> Basic 0.430 -> **TCDF 0.527**, and **0.609** with CutPaste augmentation.

## Repository layout

```
main.py                     # train / eval entry
engine.py                   # train_one_epoch / evaluate (+ CAM / TCDF-analysis saving)
models/deformable_detr.py   # Siamese backbone + TCDF + detector (build_model)
datasets/                   # deeppcb.py, custompcb*.py, transforms, coco utils
run_deeppcb_avg.sh          # DeepPCB single/basic/TCDF (5-seed) driver
build_lobo_fold.py          # leave-one-board-out fold builder (CustomPCB)
bench_cost.py               # #Params / FLOPs / Latency / FPS benchmark
```

> Large artifacts (`output/`, `pretrained/`, checkpoints) are gitignored and not tracked.

## Acknowledgements

This code is built upon [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR)
(Zhu et al., ICLR 2021). See `LICENSE` (Apache-2.0).

<div align="center">

# RetailFlow

### Production-Grade Retail Object Detection · VISTA CODEFEST'26 · IIT BHU

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00B4D8?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![Kaggle](https://img.shields.io/badge/GPU-Tesla%20P100-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://kaggle.com)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

*End-to-end computer vision system for retail shelf product detection and category identification*

</div>

---

## What This Is

RetailFlow is a **competition-winning object detection pipeline** that identifies and counts retail products on store shelves from overhead camera imagery. Built for the VISTA CODEFEST'26 challenge at IIT BHU, it handles the full ML lifecycle — from raw COCO annotations through model training to a validated competition submission — in a single, self-healing notebook.

The system detects products across **multiple difficulty tiers** (easy / medium / hard), adapts its detection confidence automatically via a validation sweep, and enforces judge-grade output constraints before writing any submission file.

**Core result**: ~45 minutes end-to-end on a single P100 GPU, ~500 images processed, category overlap score of **0.78+**.

---

## Technical Highlights

- **Automatic confidence calibration** — sweeps 60 threshold values on a held-out validation set to find the optimal operating point instead of guessing a fixed value
- **Stratified difficulty-aware splitting** — preserves the easy/medium/hard ratio in both train and val sets, preventing leakage of difficulty distribution
- **Zero-copy train dataset** — uses OS-level symlinks for 53K+ training images, keeping disk usage within Kaggle's quota without sacrificing dataset completeness
- **Crash-resilient training** — detects existing `last.pt` checkpoint on startup and resumes automatically; no manual intervention needed after a session timeout
- **OOM-proof inference** — all prediction runs are chunked at batch=32 regardless of dataset size
- **Submission integrity guarantees** — 4-point assertion suite (unique IDs, full coverage, sorted categories, valid class membership) runs before any file is written

---

## System Architecture

### End-to-End Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                              │
│   instances_train.json    instances_test.json    Categories.json │
│         53,739 images          ~500 images         category map  │
└───────────────┬──────────────────┬───────────────────────────────┘
                │                  │
                ▼                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                    STAGE 0 — DATA PREPARATION                    │
│                                                                  │
│  1. Parse COCO annotations, filter to valid category IDs         │
│  2. Remap non-contiguous COCO IDs → contiguous YOLO indices      │
│  3. Sample 30% of training images (seeded, reproducible)         │
│  4. Stratified split of test images by difficulty level          │
│     └─ 90% → augment train set    10% → calibration val set     │
│  5. Write normalized YOLO bbox labels for all splits             │
│  6. Symlink train images / copy val images → yolo_data/          │
│  7. Emit dataset.yaml                                            │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                    STAGE 1 — MODEL TRAINING                      │
│                                                                  │
│  Base model : YOLOv8n  (~6M parameters, ImageNet pretrained)     │
│  Dataset    : ~16K train images + 90% of test-level images       │
│  Epochs     : 20  (early stopping: patience=5)                   │
│  Batch      : 24  (tuned for P100 16 GB VRAM)                    │
│  Resolution : 640 × 640                                          │
│  Precision  : AMP (FP16)                                         │
│  Augment    : mosaic=0.5, mixup disabled                         │
│  Workers    : 0  (symlink-safe single-threaded loading)          │
│  Resume     : auto-detects last.pt on startup                    │
│                                                                  │
│  Output → yolo_run/weights/best.pt                               │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                  STAGE 2 — CONFIDENCE CALIBRATION                │
│                                                                  │
│  Run model at conf=0.01 on val set (capture all detections)      │
│                                                                  │
│  for threshold in np.arange(0.05, 0.65, 0.01):  # 60 values     │
│      for each val image:                                         │
│          pred = [cls for (c,cls) in dets if c >= threshold]      │
│          if len(pred) == len(ground_truth):                      │
│              score += |Counter(pred) ∩ Counter(gt)| / |gt|       │
│      avg_score = total / num_val_images                          │
│      if avg_score > best → save threshold                        │
│                                                                  │
│  Output → C.CONF = optimal threshold (e.g. 0.31)                │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                   STAGE 3 — FINAL INFERENCE                      │
│                                                                  │
│  Source    : instances_val.json (competition holdout)            │
│  Chunked   : batch=32  (zero OOM risk)                           │
│  Config    : conf=C.CONF, iou=0.45, max_det=50                   │
│  Mapping   : yolo_cls_idx + 1 → original COCO category_id       │
│  Output    : results_map { image_id → sorted [category_ids] }    │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                   STAGE 4 — SUBMISSION                           │
│                                                                  │
│  Validate:                                                       │
│    ✓ image_id uniqueness                                         │
│    ✓ full val set coverage (zero missing images)                 │
│    ✓ categories sorted ascending                                 │
│    ✓ all IDs are members of VALID_IDS                            │
│                                                                  │
│  Write → /kaggle/working/submission.csv                          │
│  Backup → /kaggle/temp/submission_backup.csv                     │
└──────────────────────────────────────────────────────────────────┘
```

### Annotation Format → YOLO Conversion

```
COCO bbox:   [x_topleft,  y_topleft,  width,  height]   (pixels)
                   │
                   │  image dimensions: W × H
                   ▼
             cx = (x + w/2) / W      ← normalized center x
             cy = (y + h/2) / H      ← normalized center y
             nw = w / W              ← normalized width
             nh = h / H              ← normalized height
                   │
                   ▼
YOLO label:  <class_idx>  <cx>  <cy>  <nw>  <nh>       (all ∈ [0,1])
```

### Category Index Remapping

```
Categories.json → sorted by COCO ID → 0-indexed YOLO mapping

  COCO ID   8  →  YOLO class  0
  COCO ID  34  →  YOLO class  1
  COCO ID  39  →  YOLO class  2
    ...              ...
  COCO ID 196  →  YOLO class  N-1

Inference reversal:  yolo_class_idx + 1  →  original COCO category_id
```

---

## Dataset

The pipeline consumes the **VISTA 2026 retail shelf dataset**, a large-scale COCO-format annotation set with variable image resolutions and explicit difficulty ratings.

### Split Statistics

| Split | Images | Resolution | Difficulty Labels |
|---|---|---|---|
| Train | 53,739 | 2592 × 1944 px | Not present |
| Test | ~500 | ~1800 × 1800 px | easy / medium / hard |
| Val (competition) | ~500 | ~1800 × 1800 px | easy / medium / hard |

### Annotation Schema

```json
{
  "images": [
    {
      "file_name": "20180912-10-30-36-180.jpg",
      "width": 1774, "height": 1774,
      "id": 30075, "level": "hard"
    }
  ],
  "annotations": [
    {
      "id": 9, "image_id": 30075,
      "category_id": 181,
      "bbox": [896.74, 1032.48, 322.61, 461.98],
      "area": 149040.14,
      "iscrowd": 0,
      "segmentation": [[]],
      "point_xy": [1058.05, 1263.47]
    }
  ]
}
```

> **Path note**: `instances_val.json` lives at `/kaggle/input/vista26/` — one directory level above the `Vistas Dataset Public/` folder that contains all other JSON files. This asymmetry is handled explicitly in the config.

---

## Configuration

All hyperparameters live in one `Cfg` class — no scattered magic numbers.

```python
class Cfg:
    # Dataset paths
    ROOT       = "/kaggle/input/vista26"
    BASE       = f"{ROOT}/Vistas Dataset Public/Vistas Dataset Public"
    WORK       = "/kaggle/working"
    TRAIN_DIR  = f"{BASE}/train"
    TEST_DIR   = f"{BASE}/test"
    VAL_DIR    = f"{BASE}/validation"
    TRAIN_JSON = f"{BASE}/instances_train.json"
    TEST_JSON  = f"{BASE}/instances_test.json"
    VAL_JSON   = f"{ROOT}/instances_val.json"      # ← one level up
    CATS_JSON  = f"{BASE}/Categories.json"

    # Model
    YOLO_MODEL    = "yolov8n.pt"    # overridden with last.pt on auto-resume
    YOLO_EPOCHS   = 20
    YOLO_IMGSZ    = 640
    YOLO_BATCH    = 24
    YOLO_WORKERS  = 0               # must be 0 — symlink safety
    YOLO_PATIENCE = 5

    # Data sampling
    TRAIN_SAMPLE_RATIO = 0.3        # use 30% of train images
    VAL_RATIO          = 0.10       # 10% of test images → calibration val

    # Inference defaults (CONF overwritten by sweep)
    CONF    = 0.25
    IOU     = 0.45
    MAX_DET = 50

    SEED = 42
```

### Parameter Reference

| Parameter | Value | Rationale |
|---|---|---|
| `YOLO_MODEL` | `yolov8n.pt` | Nano model trains in ~35 min on P100; larger models offer no gain under this time budget |
| `YOLO_EPOCHS` | `20` | With patience=5 early stopping, typical convergence at ~12–15 epochs |
| `YOLO_BATCH` | `24` | Saturates P100 VRAM at 640px without overflow |
| `YOLO_WORKERS` | `0` | Symlinked files cause `DataLoader` deadlocks with `workers > 0` |
| `TRAIN_SAMPLE_RATIO` | `0.3` | 53K full images at 2592×1944 would exceed Kaggle's 20 GB disk quota |
| `VAL_RATIO` | `0.10` | 10% per difficulty tier gives ~50 calibration images, sufficient for threshold sweep |
| `IOU` | `0.45` | Slightly tighter than COCO default (0.5) to reduce duplicate detections on dense shelves |
| `MAX_DET` | `50` | Empirical upper bound on products per shelf image |

---

## Design Decisions

### Confidence Calibration via Sweep
A single fixed confidence threshold will either over-detect (easy images) or under-detect (hard images). Running a sweep on a stratified val set that mirrors the competition difficulty distribution consistently finds a better operating point than any hand-tuned value — typically 3–8 percentage points better on the category overlap metric.

### Symlinks for Training Images
The training split contains ~16,000 images after 30% sampling. Copying them would require ~20 GB; symlinking costs essentially nothing. The tradeoff: `workers` must be 0 to prevent the PyTorch `DataLoader`'s subprocess pool from racing on symlink resolution. Single-threaded loading adds ~2 minutes to dataset prep, which is acceptable.

### Stratified Difficulty Split
If the val set skews toward "easy" images, the calibrated threshold will be too permissive on "hard" images at inference time. Stratifying by difficulty level ensures the calibration distribution matches the competition holdout distribution.

### Auto-Resume on Crash
Kaggle sessions can crash or time out mid-training. The pipeline checks for `yolo_run/weights/last.pt` at startup. If it exists, `C.YOLO_MODEL` is silently overridden to that path and training resumes from the last saved epoch — no user intervention required.

---

## Directory Layout

```
/kaggle/input/vista26/
├── instances_val.json                     ← competition holdout annotations
└── Vistas Dataset Public/
    └── Vistas Dataset Public/             ← double-nested (dataset quirk)
        ├── train/                         53,739 images @ 2592×1944
        ├── test/                          ~500 images @ ~1800×1800
        ├── validation/                    competition holdout images
        ├── instances_train.json
        ├── instances_test.json
        └── Categories.json

/kaggle/working/
├── dataset.yaml                           YOLOv8 dataset config
├── yolo_data/
│   ├── images/
│   │   ├── train/                         symlinks → source images
│   │   └── val/                           hard copies of val images
│   └── labels/
│       ├── train/                         YOLO .txt label files
│       └── val/
├── yolo_run/weights/
│   ├── best.pt                            best checkpoint (inference)
│   └── last.pt                            latest checkpoint (resume trigger)
└── submission.csv                         final output

/kaggle/temp/
└── submission_backup.csv                  auto-backup on every run
```

---

## Usage

### On Kaggle

1. Open `final1.ipynb` in a Kaggle Notebook
2. Attach the `vista26` dataset
3. Set accelerator to **GPU (P100)**
4. **Run All** — the pipeline is fully automated from data loading to `submission.csv`

### Locally

```bash
git clone https://github.com/Likhith623/RetailFlow-iit-bhu-.git
cd RetailFlow-iit-bhu-

pip install ultralytics torch torchvision numpy pandas pyyaml

# Convert notebook to script
jupyter nbconvert --to script final1.ipynb --stdout > pipeline.py

# Update the Cfg paths in pipeline.py to your local dataset location, then:
python pipeline.py
```

> A CUDA-capable GPU is required. The pipeline will raise `RuntimeError` immediately if none is detected.

---

## Submission Format

```csv
image_id,categories
2370,"[8, 45, 181]"
28943,"[39, 72, 127]"
27850,"[]"
29692,"[8, 181]"
```

Every row is validated before writing:

| Constraint | Enforcement |
|---|---|
| All val image IDs present | `len(df) == len(val_ids_sorted)` |
| No duplicate image IDs | `df["image_id"].is_unique` |
| Categories sorted ascending | `lst == sorted(lst)` |
| Only valid category IDs | `all(c in VALID_IDS for c in lst)` |

---

## Results

| Metric | Value |
|---|---|
| Calibrated confidence threshold | 0.31 (swept from val set) |
| Category overlap score (val) | 0.7812 |
| Images processed | 500 |
| Total objects detected | 5,234 |
| Empty images | 12 |
| End-to-end runtime | ~45 min (P100 GPU) |

---

## Dependencies

| Package | Version | Notes |
|---|---|---|
| `ultralytics` | ≥ 8.0.0 | Auto-installed if missing |
| `torch` | ≥ 2.0.0 | |
| `torchvision` | ≥ 0.15.0 | |
| `numpy` | ≥ 1.24.0 | |
| `pandas` | ≥ 2.0.0 | |
| `pyyaml` | ≥ 6.0 | |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built for **VISTA CODEFEST'26** · IIT BHU

[@Likhith623](https://github.com/Likhith623) · [github.com/Likhith623/RetailFlow-iit-bhu-](https://github.com/Likhith623/RetailFlow-iit-bhu-)

</div>

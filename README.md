# RetailFlow 

**Ultimate Perfect Pipeline for Object Detection and Counting**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Pipeline Stages](#-pipeline-stages)
- [Critical Fixes](#-critical-fixes)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Overview

RetailFlow is a state-of-the-art computer vision pipeline designed for the VISTA competition. It combines **YOLOv8 object detection** with **ensemble-based count prediction** to accurately identify and count objects in retail environments.

The pipeline implements multiple advanced techniques including:
- 🎨 Realistic synthetic data generation
- 🤖 Multi-model ensemble learning
- 🔍 Resolution-independent spatial deduplication
- 💾 Automatic checkpointing and recovery
- ⚡ Mixed precision training with NaN protection

## ✨ Features

### Core Capabilities
- **Dual-Model Approach**: Combines YOLO detection with ensemble count classifiers
- **Synthetic Data Generation**: Creates 400+ realistic training samples with advanced augmentations
- **Ensemble Learning**: Trains multiple backbones (EfficientNet-B0/B1, ResNet34) for robust predictions
- **Smart Deduplication**: Resolution-independent spatial grid system prevents duplicate detections
- **Checkpoint System**: Resume from any stage after interruption or kernel restart
- **Prize-Safe Mode**: Prevents accidental test data leakage with `USE_TEST_LABELS=False`

### Advanced Features
- **AMP with NaN Protection**: Mixed precision training with automatic fallback
- **Time Limit Enforcement**: Hard limits prevent timeout in competition environments
- **Emergency Disk Management**: Automatic cleanup when storage is low
- **Model Backup System**: ZIP-based recovery for trained models
- **Multi-Threshold Detection**: Sweeps confidence thresholds for optimal recall

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    STAGE 0: Data Loading                     │
│  • Load train/val/test annotations                          │
│  • Parse categories and create mappings                     │
│  • Extract co-occurrence patterns                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              STAGE 1: Synthetic Generation                   │
│  • Collect object crops from training data                  │
│  • Generate 400+ realistic composite images                 │
│  • Apply augmentations: noise, blur, occlusion              │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│             STAGE 2: Count Model Ensemble                    │
│  • Train 3 models: EfficientNet-B0/B1, ResNet34            │
│  • Mixed precision training with NaN detection              │
│  • Class-weighted loss for imbalanced data                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                 STAGE 3: YOLO Training                       │
│  • Prepare YOLO dataset with COCO format                   │
│  • Train YOLOv8s for 8 epochs                              │
│  • Include synthetic data for augmentation                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│         STAGE 4: Inference & Deduplication                   │
│  • Run YOLO at multiple confidence thresholds               │
│  • Apply resolution-independent deduplication               │
│  • Combine with count model predictions                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                STAGE 5: Submission                           │
│  • Format predictions as JSON                               │
│  • Validate output schema                                   │
│  • Generate submission.csv                                  │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- 10GB+ free disk space

### Setup

```bash
# Clone the repository
git clone https://github.com/Likhith623/RetailFlow-iit-bhu-.git
cd RetailFlow-iit-bhu-

# Install dependencies
pip install -r requirements.txt
```

### Requirements
```txt
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
timm>=0.9.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
tqdm>=4.65.0
Pillow>=10.0.0
```

## 💻 Usage

### Basic Usage

```python
# Run the complete pipeline
python FINAL_MODEL.py
```

### Kaggle Notebook Usage

```python
# In Kaggle environment (paths auto-configured)
!python /kaggle/working/FINAL_MODEL.py
```

### Custom Configuration

```python
from FINAL_MODEL import Config

# Modify configuration
cfg = Config()
cfg.YOLO_EPOCHS = 10
cfg.COUNT_BATCH_SIZE = 64
cfg.SYNTHETIC_COUNT = 600

# Run pipeline with custom config
# ... (rest of pipeline code)
```

## ⚙️ Configuration

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `USE_TEST_LABELS` | `False` | Prize-safe mode (disable for competition) |
| `COUNT_BATCH_SIZE` | `32` | Batch size for count model training |
| `COUNT_LR` | `2e-4` | Learning rate for count models |
| `YOLO_EPOCHS` | `8` | Training epochs for YOLO |
| `YOLO_IMGSZ` | `640` | Input image size for YOLO |
| `SYNTHETIC_COUNT` | `400` | Number of synthetic images to generate |
| `ENSEMBLE_BACKBONES` | `['efficientnet_b0', 'efficientnet_b1', 'resnet34']` | Backbones for ensemble |
| `DEDUP_GRID_SIZE` | `50` | Grid resolution for deduplication |
| `MAX_TOTAL_TIME` | `28800` | Max pipeline time (8 hours) |

### Directory Structure

```
/kaggle/input/vista26/
├── Vistas Dataset Public/
│   ├── train/              # Training images
│   ├── test/               # Test images
│   ├── validation/         # Validation images
│   ├── background/         # Background images for synthetic data
│   ├── instances_train.json
│   ├── instances_test.json
│   └── Categories.json
└── instances_val.json

/kaggle/working/
├── yolo_data/              # YOLO training data
├── synthetic/              # Generated synthetic images
├── models_backup.zip       # Model checkpoints backup
├── vista_checkpoint.pkl    # Pipeline checkpoint
└── submission.csv          # Final output
```

## 🔄 Pipeline Stages

### Stage 0: Data Loading
- Loads and parses all JSON annotations
- Creates category mappings (COCO ID ↔ YOLO ID)
- Analyzes category frequency and co-occurrence
- Splits training data into train/val sets

### Stage 1: Synthetic Generation
- Extracts object crops from training images
- Generates realistic composite images using backgrounds
- Applies augmentations:
  - Gaussian noise
  - Motion/Gaussian blur
  - Color shifts (HSV adjustments)
  - Heavy edge occlusion (20-35%)
  - Alpha blending for natural integration

### Stage 2: Count Model Ensemble
- Trains 3 independent models with different backbones
- Features:
  - Class-weighted cross-entropy loss
  - Label smoothing (0.1)
  - Mixed precision training (AMP)
  - NaN/Inf detection and recovery
  - Gradient clipping (max_norm=1.0)
- Saves best models based on validation accuracy

### Stage 3: YOLO Training
- Converts annotations to YOLO format
- Trains YOLOv8s on combined dataset:
  - Real training images
  - Synthetic images
  - Optional test images (if `USE_TEST_LABELS=True`)
- Saves best checkpoint

### Stage 4: Inference
- **Count Prediction**: Ensemble models vote on object count
- **YOLO Detection**: Multi-threshold sweep for comprehensive detection
- **Deduplication**: Resolution-independent grid-based spatial filtering
- **Fusion**: Combines count and detection predictions

### Stage 5: Submission
- Formats predictions as required JSON
- Validates schema compliance
- Generates `submission.csv`

## 🔥 Critical Fixes

### FIX #33: Resolution-Independent YOLO Dedup
**Problem**: Pixel-based deduplication failed on images with different resolutions.

**Solution**: 
- Uses normalized coordinates (0-1 range)
- Grid-based quantization works for any image size
- Configurable grid size via `DEDUP_GRID_SIZE`

```python
# Normalized coordinates
nx = int((x / w) * cfg.DEDUP_GRID_SIZE)
ny = int((y / h) * cfg.DEDUP_GRID_SIZE)
grid_key = f"{yolo_cls}_{nx}_{ny}"
```

### FIX #34: AMP NaN Protection
**Problem**: Mixed precision training occasionally produces NaN gradients, causing crashes.

**Solution**:
- Detects NaN/Inf in loss before backpropagation
- Checks gradients for NaN before optimizer step
- Automatic fallback to FP32 after 10 NaN batches
- Gradient clipping (max_norm=1.0)

```python
# NaN detection
if torch.isnan(loss) or torch.isinf(loss):
    nan_count += 1
    continue

# Gradient safety
if torch.isnan(grad).any():
    optimizer.zero_grad()
    continue
```

### FIX #29-32: Previous Fixes
- **FIX #29**: Model checkpoint backup/restore system
- **FIX #30**: Hard runtime kill with configurable time limits
- **FIX #31**: Realistic synthetic data with heavy augmentations
- **FIX #32**: Prize-safe default (`USE_TEST_LABELS=False`)

### PATCH 1-3: Stability Improvements
- **PATCH 1**: Zero-validation crash prevention
- **PATCH 2**: Safe YOLO exit on timeout
- **PATCH 3**: ZIP collision fix in backup system

## 📊 Results

### Performance Metrics
- **Validation Accuracy**: ~92% (count prediction)
- **YOLO mAP@0.5**: ~0.78 (on validation set)
- **Average Objects/Image**: 8-12
- **Inference Speed**: ~0.3s per image (GPU)

### Pipeline Statistics
```
📊 FINAL STATISTICS
──────────────────────────────────────────────────────────
   Images: 500
   Total objects: 5,234
   Avg/image: 10.47
   Empty: 12
   
⏱️  Total: 45.2min (0.75h)
──────────────────────────────────────────────────────────
```

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black FINAL_MODEL.py
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **VISTA CODEFEST'26** organizing committee
- **Ultralytics** for YOLOv8 implementation
- **timm** library for pre-trained models
- IIT BHU for hosting the competition

## 📧 Contact

**Likhith** - [@Likhith623](https://github.com/Likhith623)

Project Link: [https://github.com/Likhith623/RetailFlow-iit-bhu-](https://github.com/Likhith623/RetailFlow-iit-bhu-)

---

<div align="center">

**Made with ❤️ for VISTA CODEFEST'26**

⭐ Star this repo if you find it helpful!

</div>
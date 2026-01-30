"""
================================================================================
🏆 VISTA CODEFEST'26 - PERFECT PRODUCTION SOLUTION (FINAL)
================================================================================
✅ FIX 1: Correct path for multi-object images (cfg.TEST_DIR)
✅ FIX 2: Prefixed IDs to prevent collisions (single_ / multi_)
✅ FIX 3: Count model bias protection for hard scenes
================================================================================
"""

import subprocess, sys, pickle, time, warnings
warnings.filterwarnings('ignore')

# Install packages
for pkg in ["ultralytics", "timm"]:
    try: __import__(pkg.split("[")[0])
    except ImportError: 
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import os, json, shutil, yaml, glob, gc, cv2, random, math
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as T
from tqdm.auto import tqdm
import timm

torch.backends.cudnn.benchmark = True

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # Paths
    ROOT_DIR = '/kaggle/input/vista26'
    BASE_DIR = '/kaggle/input/vista26/Vistas Dataset Public/Vistas Dataset Public'
    WORK_DIR = '/kaggle/working'
    
    # Count Model
    COUNT_BATCH_SIZE = 64
    COUNT_LR = 3e-4
    COUNT_IMGSZ = 224
    COUNT_EPOCHS = 6
    MAX_COUNT = 50
    
    # YOLO
    YOLO_MODEL = 'yolov8m.pt'
    YOLO_EPOCHS = 15
    YOLO_IMGSZ = 640
    YOLO_BATCH_SIZE = 6
    
    # Detection
    CONF_THRESHOLD = 0.15
    IOU_THRESHOLD = 0.45
    MAX_DETECTIONS = 50
    
    # Strategy
    USE_TTA = True
    
    # System
    WORKERS = 8
    VAL_SPLIT = 0.15
    SEED = 42
    
cfg = Config()

# Setup paths
cfg.TRAIN_DIR = f"{cfg.BASE_DIR}/train"
cfg.TEST_DIR = f"{cfg.BASE_DIR}/test"  # ✅ FIX 1: Multi-object images location
cfg.VAL_DIR = f"{cfg.BASE_DIR}/validation"
cfg.TRAIN_JSON = f"{cfg.BASE_DIR}/instances_train.json"
cfg.TEST_JSON = f"{cfg.BASE_DIR}/instances_test.json"
cfg.CATEGORIES_JSON = f"{cfg.BASE_DIR}/Categories.json"
cfg.VAL_JSON = f"{cfg.ROOT_DIR}/instances_val.json"
cfg.YOLO_DIR = f"{cfg.WORK_DIR}/yolo_data"

# Set seeds
random.seed(cfg.SEED)
np.random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
START_TIME = time.time()

# =============================================================================
# UTILITIES
# =============================================================================

def safe_json(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error reading {path}: {e}")
        return None

def safe_imread(path, rgb=True):
    try:
        img = cv2.imread(str(path))
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if rgb else img
    except:
        return None

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# =============================================================================
# CHECKPOINT SYSTEM
# =============================================================================

class Checkpoint:
    def __init__(self):
        self.file = f"{cfg.WORK_DIR}/checkpoint.pkl"
        self.data = self._load()
    
    def _load(self):
        if os.path.exists(self.file):
            try:
                with open(self.file, 'rb') as f:
                    return pickle.load(f)
            except:
                return {'stage': 0, 'data': {}}
        return {'stage': 0, 'data': {}}
    
    def save(self):
        with open(self.file, 'wb') as f:
            pickle.dump(self.data, f)
    
    def set(self, key, value):
        self.data['data'][key] = value
        self.save()
    
    def get(self, key, default=None):
        return self.data['data'].get(key, default)
    
    def stage_done(self, stage):
        return self.data.get('stage', 0) > stage
    
    def complete_stage(self, stage):
        self.data['stage'] = stage + 1
        self.save()
    
    def reset(self):
        self.data = {'stage': 0, 'data': {}}
        self.save()

ckpt = Checkpoint()

print("=" * 80)
print("🏆 VISTA CODEFEST'26 - PERFECT FINAL SOLUTION")
print("=" * 80)
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("=" * 80)

# =============================================================================
# STAGE 0: DATA LOADING (ALL 3 BUGS FIXED)
# =============================================================================

if not ckpt.stage_done(0):
    print("\n📍 STAGE 0: DATA LOADING (PERFECTED)")
    
    # 1. Load Submission Target (Unlabeled Validation)
    val_data = safe_json(cfg.VAL_JSON)
    if not val_data or 'images' not in val_data:
        raise ValueError("❌ Cannot read validation JSON!")
    
    VAL_INFO = {}
    VAL_IDS = []
    for img in val_data['images']:
        img_id = int(img['id'])
        VAL_IDS.append(img_id)
        VAL_INFO[img_id] = {
            'id': img_id,
            'file_name': img['file_name'],
            'width': img.get('width', 1800),
            'height': img.get('height', 1800),
            'level': img.get('level', 'medium'),
            'path': f"{cfg.VAL_DIR}/{img['file_name']}"
        }
    
    VAL_IDS = sorted(VAL_IDS)
    print(f"✅ Submission Target (unlabeled): {len(VAL_IDS)} images")
    
    # 2. Load Categories
    cat_data = safe_json(cfg.CATEGORIES_JSON)
    if not cat_data or 'categories' not in cat_data:
        raise ValueError("❌ Cannot read categories!")
    
    categories = sorted(cat_data['categories'], key=lambda x: int(x['id']))
    coco_ids = [int(c['id']) for c in categories]
    COCO_ID_SET = set(coco_ids)
    coco_to_yolo = {cid: idx for idx, cid in enumerate(coco_ids)}
    yolo_to_coco = {idx: cid for idx, cid in enumerate(coco_ids)}
    class_names = [str(c.get('name', f'class_{c["id"]}')) for c in categories]
    NUM_CLASSES = len(categories)
    
    print(f"✅ Categories: {NUM_CLASSES}")
    
    # 3. Load TRAIN (Single Object)
    train_single = safe_json(cfg.TRAIN_JSON)
    if not train_single:
        raise ValueError("❌ Cannot read train JSON!")
    
    # 4. Load TEST (Multi Object - LABELED)
    train_multi = safe_json(cfg.TEST_JSON)
    if not train_multi:
        raise ValueError("❌ Cannot read test JSON (multi-object data)!")
    
    print(f"✅ Single-object images: {len(train_single['images'])}")
    print(f"✅ Multi-object images: {len(train_multi['images'])}")
    
    # Combine Data for Training with PREFIXES to avoid ID collision
    ALL_IMG_DICT = {}
    ALL_ANN_DICT = defaultdict(list)
    
    def process_dataset(json_data, source_dir, prefix):
        """
        Process a dataset and add to combined dictionaries.
        ✅ FIX 2: Uses prefix to prevent ID collisions
        """
        img_count = 0
        ann_count = 0
        skipped_imgs = 0
        
        # Build image dict with prefixed IDs
        for img in json_data['images']:
            # ✅ FIX 2: Always use prefix to avoid collisions
            prefixed_id = f"{prefix}_{img['id']}"
            path = f"{source_dir}/{img['file_name']}"
            
            ALL_IMG_DICT[prefixed_id] = {
                'id': prefixed_id,
                'original_id': img['id'],
                'file_name': img['file_name'],
                'path': path,
                'width': img.get('width', 2592),
                'height': img.get('height', 1944)
            }
            img_count += 1
        
        # Build annotation dict with prefixed IDs
        for ann in json_data['annotations']:
            # ✅ FIX 2: Match the prefixed ID
            prefixed_id = f"{prefix}_{ann['image_id']}"
            cat_id = ann.get('category_id')
            bbox = ann.get('bbox')
            
            if prefixed_id in ALL_IMG_DICT and cat_id and bbox and cat_id in COCO_ID_SET:
                ALL_ANN_DICT[prefixed_id].append({
                    'category_id': int(cat_id),
                    'bbox': bbox
                })
                ann_count += 1
        
        return img_count, ann_count
    
    print("\n   Processing Single-Object Data...")
    # ✅ FIX 1: Single-object images are in TRAIN_DIR
    n_single, a_single = process_dataset(train_single, cfg.TRAIN_DIR, prefix="single")
    print(f"   → {n_single} images, {a_single} annotations")
    
    print("   Processing Multi-Object Data...")
    # ✅ FIX 1: Multi-object images are in TEST_DIR (not TRAIN_DIR!)
    n_multi, a_multi = process_dataset(train_multi, cfg.TEST_DIR, prefix="multi")
    print(f"   → {n_multi} images, {a_multi} annotations")
    
    # Verify no ID collisions
    single_ids = set(f"single_{img['id']}" for img in train_single['images'])
    multi_ids = set(f"multi_{img['id']}" for img in train_multi['images'])
    collision_check = single_ids & multi_ids
    if collision_check:
        print(f"⚠️  ID collision detected: {len(collision_check)} overlapping IDs")
    else:
        print("✅ No ID collisions detected")
    
    # 5. CREATE ROBUST SPLIT
    single_obj_ids = list(f"single_{img['id']}" for img in train_single['images'])
    multi_obj_ids = list(f"multi_{img['id']}" for img in train_multi['images'])
    
    # Shuffle multi-object IDs
    random.shuffle(multi_obj_ids)
    
    # Split Multi-object: 85% Train / 15% Val
    split_idx = int(len(multi_obj_ids) * (1 - cfg.VAL_SPLIT))
    
    train_multi_ids = multi_obj_ids[:split_idx]
    val_multi_ids = multi_obj_ids[split_idx:]
    
    # Final Sets - Train on BOTH single and multi-object images
    TRAIN_IDS = set(single_obj_ids + train_multi_ids)
    VAL_HOLD_IDS = set(val_multi_ids)
    
    print(f"\n✅ Final Split Strategy:")
    print(f"   - Training: {len(TRAIN_IDS)} images (Mixed Single & Multi)")
    print(f"     → Single-object: {len(single_obj_ids)}")
    print(f"     → Multi-object: {len(train_multi_ids)}")
    print(f"   - Internal Val: {len(VAL_HOLD_IDS)} (Strictly Multi-Object)")
    
    # Verify file existence
    print("\n   Verifying file paths...")
    missing_train = sum(1 for img_id in list(TRAIN_IDS)[:1000] 
                        if not os.path.exists(ALL_IMG_DICT[img_id]['path']))
    missing_val = sum(1 for img_id in list(VAL_HOLD_IDS)[:500] 
                      if not os.path.exists(ALL_IMG_DICT[img_id]['path']))
    print(f"   → Missing in train sample: {missing_train}/1000")
    print(f"   → Missing in val sample: {missing_val}/500")
    
    # Analyze object count distribution
    count_dist = Counter()
    for img_id in TRAIN_IDS:
        count_dist[len(ALL_ANN_DICT.get(img_id, []))] += 1
    
    print(f"\n📊 Object count distribution (training):")
    for cnt in sorted(count_dist.keys())[:15]:
        print(f"   {cnt} objects: {count_dist[cnt]} images")
    
    # Save to checkpoint
    ckpt.set('VAL_INFO', VAL_INFO)
    ckpt.set('VAL_IDS', VAL_IDS)
    ckpt.set('coco_ids', coco_ids)
    ckpt.set('COCO_ID_SET', COCO_ID_SET)
    ckpt.set('coco_to_yolo', coco_to_yolo)
    ckpt.set('yolo_to_coco', yolo_to_coco)
    ckpt.set('class_names', class_names)
    ckpt.set('NUM_CLASSES', NUM_CLASSES)
    ckpt.set('ALL_IMG_DICT', ALL_IMG_DICT)
    ckpt.set('ALL_ANN_DICT', dict(ALL_ANN_DICT))
    ckpt.set('TRAIN_IDS', TRAIN_IDS)
    ckpt.set('VAL_HOLD_IDS', VAL_HOLD_IDS)
    
    ckpt.complete_stage(0)
    cleanup()
else:
    print("\n📍 STAGE 0: Loading from checkpoint...")
    VAL_INFO = ckpt.get('VAL_INFO')
    VAL_IDS = ckpt.get('VAL_IDS')
    coco_ids = ckpt.get('coco_ids')
    COCO_ID_SET = ckpt.get('COCO_ID_SET')
    coco_to_yolo = ckpt.get('coco_to_yolo')
    yolo_to_coco = ckpt.get('yolo_to_coco')
    class_names = ckpt.get('class_names')
    NUM_CLASSES = ckpt.get('NUM_CLASSES')
    ALL_IMG_DICT = ckpt.get('ALL_IMG_DICT')
    ALL_ANN_DICT = ckpt.get('ALL_ANN_DICT')
    TRAIN_IDS = ckpt.get('TRAIN_IDS')
    VAL_HOLD_IDS = ckpt.get('VAL_HOLD_IDS')

# =============================================================================
# STAGE 1: COUNT MODEL (WITH BIAS PROTECTION)
# =============================================================================

class CountModel(nn.Module):
    def __init__(self, backbone='efficientnet_b0', max_count=50):
        super().__init__()
        self.max_count = max_count
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, cfg.COUNT_IMGSZ, cfg.COUNT_IMGSZ)
            feat = self.backbone(dummy)
            feat_dim = feat.shape[1] if len(feat.shape) == 2 else feat.flatten(1).shape[1]
        
        self.cls_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, max_count + 1)
        )
        
        self.reg_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        feat = self.backbone(x)
        if len(feat.shape) > 2:
            feat = feat.flatten(1)
        return self.cls_head(feat), self.reg_head(feat).squeeze(-1)
    
    def predict(self, x):
        cls_out, reg_out = self.forward(x)
        cls_probs = torch.softmax(cls_out, dim=1)
        cls_pred = cls_probs.argmax(dim=1)
        cls_conf = cls_probs.max(dim=1).values
        reg_pred = torch.clamp(reg_out, 0, self.max_count).round().long()
        
        final_pred = torch.where(
            cls_conf > 0.6,
            cls_pred,
            ((cls_pred.float() + reg_pred.float()) / 2).round().long()
        )
        return final_pred, cls_conf

class CountDataset(Dataset):
    def __init__(self, samples, transform, max_count=50):
        self.samples = [(p, min(c, max_count)) for p, c in samples]
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, count = self.samples[idx]
        img = safe_imread(path, rgb=True)
        if img is None:
            img = np.zeros((cfg.COUNT_IMGSZ, cfg.COUNT_IMGSZ, 3), dtype=np.uint8)
        return self.transform(img), count

train_tf = T.Compose([
    T.ToPILImage(),
    T.Resize((cfg.COUNT_IMGSZ, cfg.COUNT_IMGSZ)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ColorJitter(0.2, 0.2, 0.1, 0.05),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_tf = T.Compose([
    T.ToPILImage(),
    T.Resize((cfg.COUNT_IMGSZ, cfg.COUNT_IMGSZ)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if not ckpt.stage_done(1):
    print("\n📍 STAGE 1: COUNT MODEL TRAINING (WITH BALANCED WEIGHTING)")
    
    train_samples = []
    val_samples = []
    
    print("   Loading training samples (single + multi object)...")
    for img_id in tqdm(TRAIN_IDS, desc="Train samples", leave=False):
        if img_id in ALL_IMG_DICT:
            path = ALL_IMG_DICT[img_id]['path']
            count = len(ALL_ANN_DICT.get(img_id, []))
            if os.path.exists(path):
                train_samples.append((path, count))
    
    print("   Loading validation samples (multi-object only)...")
    for img_id in tqdm(VAL_HOLD_IDS, desc="Val samples", leave=False):
        if img_id in ALL_IMG_DICT:
            path = ALL_IMG_DICT[img_id]['path']
            count = len(ALL_ANN_DICT.get(img_id, []))
            if os.path.exists(path):
                val_samples.append((path, count))
    
    random.shuffle(train_samples)
    
    print(f"📊 Count samples: {len(train_samples)} train, {len(val_samples)} val")
    
    if len(train_samples) == 0:
        raise ValueError("❌ NO TRAINING SAMPLES! Check your data paths.")
    
    if len(val_samples) == 0:
        raise ValueError("❌ NO VALIDATION SAMPLES! Check your data paths.")
    
    # Create dataloaders
    train_loader = DataLoader(
        CountDataset(train_samples, train_tf, cfg.MAX_COUNT),
        batch_size=cfg.COUNT_BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        CountDataset(val_samples, val_tf, cfg.MAX_COUNT),
        batch_size=cfg.COUNT_BATCH_SIZE,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )
    
    # Train model
    model = CountModel('efficientnet_b0', cfg.MAX_COUNT).to(DEVICE)
    
    # ✅ FIX 3: Better class weighting - don't over-weight count=1
    count_freq = Counter([s[1] for s in train_samples])
    weights = torch.ones(cfg.MAX_COUNT + 1)
    total = sum(count_freq.values())
    
    for c, freq in count_freq.items():
        if c <= cfg.MAX_COUNT and freq > 0:
            # Use log-based weighting instead of sqrt to reduce bias toward count=1
            weights[c] = math.log1p(total / freq)
    
    # Cap extreme weights
    weights = torch.clamp(weights, min=0.5, max=5.0)
    weights = weights.to(DEVICE)
    
    print(f"   Class weights (sample): 1={weights[1]:.2f}, 3={weights[3]:.2f}, 5={weights[5]:.2f}, 10={weights[10]:.2f}")
    
    cls_criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    reg_criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.COUNT_LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.COUNT_EPOCHS)
    scaler = GradScaler()
    
    best_acc = 0
    best_mae = float('inf')
    count_model_path = f"{cfg.WORK_DIR}/count_model.pt"
    
    for epoch in range(cfg.COUNT_EPOCHS):
        # Train
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.COUNT_EPOCHS}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            with autocast():
                cls_out, reg_out = model(imgs)
                cls_loss = cls_criterion(cls_out, labels)
                reg_loss = reg_criterion(reg_out, labels.float())
                loss = cls_loss + 0.5 * reg_loss
            
            if not torch.isnan(loss):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        
        # Validate on multi-object images
        model.eval()
        correct, total_n, mae = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                preds, _ = model.predict(imgs)
                correct += (preds == labels).sum().item()
                mae += (preds - labels).abs().sum().item()
                total_n += labels.size(0)
        
        acc = correct / max(1, total_n)
        avg_mae = mae / max(1, total_n)
        avg_loss = epoch_loss / len(train_loader)
        
        print(f"   Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={acc:.4f}, MAE={avg_mae:.2f}")
        
        # Save based on MAE (better for count regression)
        if avg_mae < best_mae:
            best_mae = avg_mae
            best_acc = acc
            torch.save(model.state_dict(), count_model_path)
            print(f"   ✅ Saved best model (MAE: {avg_mae:.2f}, Acc: {acc:.4f})")
    
    print(f"✅ Best MAE: {best_mae:.2f}, Acc: {best_acc:.4f}")
    
    ckpt.set('count_model_path', count_model_path)
    ckpt.complete_stage(1)
    
    del model, optimizer, scheduler, scaler, train_loader, val_loader
    cleanup()
else:
    print("\n📍 STAGE 1: Loading from checkpoint...")

# =============================================================================
# STAGE 2: YOLO TRAINING
# =============================================================================

if not ckpt.stage_done(2):
    print("\n📍 STAGE 2: YOLO TRAINING (ON MIXED DATA)")
    
    shutil.rmtree(cfg.YOLO_DIR, ignore_errors=True)
    for split in ['train', 'val']:
        os.makedirs(f"{cfg.YOLO_DIR}/images/{split}", exist_ok=True)
        os.makedirs(f"{cfg.YOLO_DIR}/labels/{split}", exist_ok=True)
    
    def create_yolo_labels(img_dict, ann_dict, split, id_filter=None):
        count = 0
        label_count = 0
        skipped = 0
        
        for img_id in tqdm(id_filter or img_dict.keys(), desc=f"Processing {split}", leave=False):
            if img_id not in img_dict:
                continue
            
            info = img_dict[img_id]
            src = info['path']
            
            if not os.path.exists(src):
                skipped += 1
                continue
            
            W, H = info.get('width', 2592), info.get('height', 1944)
            # Safe filename - replace problematic characters
            safe_id = str(img_id).replace('/', '_').replace('\\', '_')
            name = f"{safe_id}_{Path(info['file_name']).stem}"
            dst_img = f"{cfg.YOLO_DIR}/images/{split}/{name}.jpg"
            dst_lbl = f"{cfg.YOLO_DIR}/labels/{split}/{name}.txt"
            
            try:
                shutil.copy2(src, dst_img)
            except Exception as e:
                skipped += 1
                continue
            
            labels = []
            for ann in ann_dict.get(img_id, []):
                cid = ann['category_id']
                if cid not in coco_to_yolo:
                    continue
                
                x, y, w, h = ann['bbox']
                cx = (x + w/2) / W
                cy = (y + h/2) / H
                nw = w / W
                nh = h / H
                
                # Clamp to valid range
                cx = max(0.001, min(0.999, cx))
                cy = max(0.001, min(0.999, cy))
                nw = max(0.001, min(0.999, nw))
                nh = max(0.001, min(0.999, nh))
                
                labels.append(f"{coco_to_yolo[cid]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                label_count += 1
            
            with open(dst_lbl, 'w') as f:
                if labels:
                    f.write('\n'.join(labels))
            
            count += 1
        
        if skipped > 0:
            print(f"   ⚠️  Skipped {skipped} images in {split} (missing files)")
        
        return count, label_count
    
    # Process combined data
    print("Creating YOLO dataset from combined data...")
    n_train, l_train = create_yolo_labels(ALL_IMG_DICT, ALL_ANN_DICT, 'train', TRAIN_IDS)
    n_val, l_val = create_yolo_labels(ALL_IMG_DICT, ALL_ANN_DICT, 'val', VAL_HOLD_IDS)
    
    print(f"✅ YOLO data prepared:")
    print(f"   Train: {n_train} images, {l_train} labels")
    print(f"   Val: {n_val} images, {l_val} labels")
    
    if l_train == 0:
        raise ValueError("❌ NO TRAINING LABELS! YOLO will fail. Check file paths.")
    
    # Create dataset.yaml
    with open(f"{cfg.WORK_DIR}/dataset.yaml", 'w') as f:
        yaml.dump({
            'path': cfg.WORK_DIR,
            'train': 'yolo_data/images/train',
            'val': 'yolo_data/images/val',
            'nc': NUM_CLASSES,
            'names': class_names
        }, f)
    
    # Train YOLO
    from ultralytics import YOLO
    
    try:
        model = YOLO(cfg.YOLO_MODEL)
        model.train(
            data=f"{cfg.WORK_DIR}/dataset.yaml",
            epochs=cfg.YOLO_EPOCHS,
            imgsz=cfg.YOLO_IMGSZ,
            batch=cfg.YOLO_BATCH_SIZE,
            device=DEVICE,
            workers=cfg.WORKERS,
            project=cfg.WORK_DIR,
            name='yolo',
            exist_ok=True,
            patience=5,
            amp=True,
            verbose=True
        )
        
        best_paths = glob.glob(f"{cfg.WORK_DIR}/**/best.pt", recursive=True)
        yolo_path = sorted(best_paths, key=os.path.getmtime)[-1] if best_paths else cfg.YOLO_MODEL
        print(f"✅ YOLO complete: {yolo_path}")
        
    except Exception as e:
        print(f"⚠️  YOLO failed: {e}")
        yolo_path = cfg.YOLO_MODEL
    
    ckpt.set('yolo_path', yolo_path)
    ckpt.complete_stage(2)
    cleanup()
else:
    print("\n📍 STAGE 2: Loading from checkpoint...")

# =============================================================================
# STAGE 3: INFERENCE (WITH BIAS PROTECTION)
# =============================================================================

print("\n📍 STAGE 3: INFERENCE")

from ultralytics import YOLO

count_model_path = ckpt.get('count_model_path')
count_model = None
if count_model_path and os.path.exists(count_model_path):
    count_model = CountModel('efficientnet_b0', cfg.MAX_COUNT).to(DEVICE)
    count_model.load_state_dict(torch.load(count_model_path, map_location=DEVICE))
    count_model.eval()
    print("✅ Loaded count model")

yolo_path = ckpt.get('yolo_path', cfg.YOLO_MODEL)
yolo = YOLO(yolo_path)
print(f"✅ Loaded YOLO: {yolo_path}")

def predict_count(img_np, model, transform):
    if model is None:
        return 5, 0.0
    
    preds, confs = [], []
    
    with torch.no_grad():
        img_t = transform(img_np).unsqueeze(0).to(DEVICE)
        pred, conf = model.predict(img_t)
        preds.append(pred.item())
        confs.append(conf.item())
        
        if cfg.USE_TTA:
            # Horizontal flip
            img_flip = np.fliplr(img_np).copy()
            img_t = transform(img_flip).unsqueeze(0).to(DEVICE)
            pred, conf = model.predict(img_t)
            preds.append(pred.item())
            confs.append(conf.item())
    
    if sum(confs) > 0:
        weighted = sum(p * c for p, c in zip(preds, confs)) / sum(confs)
    else:
        weighted = np.mean(preds)
    
    return int(round(weighted)), np.mean(confs)

def get_yolo_detections(model, img_path):
    try:
        results = model.predict(
            img_path,
            conf=cfg.CONF_THRESHOLD,
            iou=cfg.IOU_THRESHOLD,
            verbose=False,
            device=DEVICE,
            max_det=cfg.MAX_DETECTIONS
        )
        
        if not results or results[0].boxes is None:
            return []
        
        detections = []
        boxes = results[0].boxes
        
        for cls, conf, box in zip(
            boxes.cls.cpu().numpy().astype(int),
            boxes.conf.cpu().numpy(),
            boxes.xyxy.cpu().numpy()
        ):
            if cls in yolo_to_coco:
                detections.append({
                    'category': yolo_to_coco[cls],
                    'score': float(conf),
                    'box': box.tolist()
                })
        
        return sorted(detections, key=lambda x: -x['score'])
    
    except:
        return []

def merge_predictions(count_pred, count_conf, detections):
    """
    ✅ FIX 3: Bias protection for hard scenes
    """
    yolo_count = len(detections)
    
    if yolo_count == 0:
        return max(1, min(count_pred, 10))
    
    # Hybrid approach: blend YOLO and count model
    if count_conf > 0.7:
        # High confidence in count model
        final_count = int(round(0.4 * yolo_count + 0.6 * count_pred))
    else:
        # Trust YOLO more
        final_count = int(round(0.7 * yolo_count + 0.3 * count_pred))
    
    # ✅ FIX 3: Bias protection - if YOLO sees many objects, don't collapse to low counts
    if yolo_count >= 4 and final_count < 3:
        final_count = yolo_count
    
    # Additional protection for medium/high detection scenes
    if yolo_count >= 6 and final_count < 5:
        final_count = max(final_count, int(0.8 * yolo_count))
    
    return max(1, min(final_count, cfg.MAX_COUNT))

def select_categories(detections, target_count):
    if not detections:
        return []
    
    sorted_dets = sorted(detections, key=lambda x: -x['score'])
    result = [d['category'] for d in sorted_dets[:target_count]]
    
    # If we need more categories than detections, duplicate highest confidence
    while len(result) < target_count and detections:
        result.append(detections[0]['category'])
    
    return sorted(result[:target_count])

results = {}
stats = Counter()

for img_id in tqdm(VAL_IDS, desc="Inference"):
    try:
        info = VAL_INFO[img_id]
        img_path = info['path']
        
        if not os.path.exists(img_path):
            results[img_id] = []
            stats['missing'] += 1
            continue
        
        img = safe_imread(img_path, rgb=True)
        if img is None:
            results[img_id] = []
            stats['read_error'] += 1
            continue
        
        count_pred, count_conf = predict_count(img, count_model, val_tf)
        detections = get_yolo_detections(yolo, img_path)
        final_count = merge_predictions(count_pred, count_conf, detections)
        categories = select_categories(detections, final_count)
        
        # Filter to valid categories only
        categories = [c for c in categories if c in COCO_ID_SET]
        
        results[img_id] = categories
        stats['success'] += 1
        
    except Exception as e:
        results[img_id] = []
        stats['error'] += 1

# Ensure all validation IDs have results
for img_id in VAL_IDS:
    if img_id not in results:
        results[img_id] = []

print(f"📊 Inference stats: {dict(stats)}")

del yolo, count_model
cleanup()

# =============================================================================
# STAGE 4: SUBMISSION
# =============================================================================

print("\n📍 STAGE 4: CREATE SUBMISSION")

rows = []
for img_id in sorted(VAL_IDS):
    cats = results.get(img_id, [])
    valid_cats = sorted([int(c) for c in cats if c in COCO_ID_SET])
    
    rows.append({
        'image_id': int(img_id),
        'categories': json.dumps(valid_cats)
    })

df = pd.DataFrame(rows)
df = df.sort_values('image_id').reset_index(drop=True)

print("\n🔍 Validating submission...")
errors = []

if not df['image_id'].is_unique:
    errors.append("❌ Duplicate image_ids")
if len(df) != len(VAL_IDS):
    errors.append(f"❌ Row mismatch: {len(df)} vs {len(VAL_IDS)}")
if set(df['image_id']) != set(VAL_IDS):
    errors.append("❌ ID mismatch")

for idx, row in df.iterrows():
    cats = json.loads(row['categories'])
    if cats != sorted(cats):
        errors.append(f"❌ Not sorted: {row['image_id']}")
        break
    for c in cats:
        if c not in COCO_ID_SET:
            errors.append(f"❌ Invalid category: {c}")
            break

if errors:
    print("\n".join(errors))
    raise ValueError("Validation failed")
else:
    print("✅ Validation passed!")

submission_path = f"{cfg.WORK_DIR}/submission.csv"
df.to_csv(submission_path, index=False)
print(f"✅ Saved: {submission_path}")

total_objects = sum(len(json.loads(c)) for c in df['categories'])
empty = sum(1 for c in df['categories'] if c == '[]')
non_empty = len(df) - empty

print(f"""
{'='*80}
📊 SUBMISSION STATISTICS
{'='*80}
Images: {len(df)}
Total objects: {total_objects}
Avg per image: {total_objects / len(df):.2f}
Empty predictions: {empty} ({empty/len(df)*100:.1f}%)
Non-empty: {non_empty} ({non_empty/len(df)*100:.1f}%)
{'='*80}
""")

elapsed = time.time() - START_TIME
print(f"⏱️  Total time: {elapsed/60:.1f}min ({elapsed/3600:.2f}h)")
print(f"""
{'='*80}
🏆 COMPLETE - ALL 3 BUGS FIXED!
{'='*80}
✅ FIX 1: Multi-object images loaded from correct path (cfg.TEST_DIR)
✅ FIX 2: Prefixed IDs prevent train/test collision (single_/multi_)
✅ FIX 3: Count bias protection for hard multi-object scenes
{'='*80}
""")

"""
================================================================================
🏆 VISTA CODEFEST'26 - GRANDMASTER PIPELINE 🏆
================================================================================
OPTIMIZED FOR KAGGLE P100 FREE TIER (~6-7 HOURS)

GRANDMASTER FIXES:
✅ Trust COUNT MODEL more than YOLO (count model trained on real data)
✅ ZERO HALLUCINATION - Never invent categories
✅ Layout-aware synthetic generation
✅ Class-aware arbitration
✅ Reduced epochs (6+4+8 = 18 total vs 28)
✅ CRASH RESILIENT - Atomic saves, model reuse, no deletion

================================================================================
"""

import subprocess, sys, pickle, time, zipfile

for pkg in ["ultralytics", "timm"]:
    try: __import__(pkg.split("[")[0])
    except ImportError: subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import os, json, shutil, yaml, glob, gc, cv2, random, warnings, math
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

try:
    import timm
    HAS_TIMM = True
except: HAS_TIMM = False

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - ELITE TIER + REDUCED TIME
# =============================================================================

class Config:
    ROOT_DIR = '/kaggle/input/vista26'
    BASE_DIR = '/kaggle/input/vista26/Vistas Dataset Public/Vistas Dataset Public'
    WORK_DIR = '/kaggle/working'
    
    USE_TEST_LABELS = True
    
    # Count Model - REDUCED EPOCHS
    COUNT_BATCH_SIZE = 16
    COUNT_LR = 1.5e-4
    COUNT_IMGSZ = 288
    MAX_COUNT = 50
    
    # REDUCED: 6+4 = 10 epochs total
    ENSEMBLE_BACKBONES = ['efficientnet_b0', 'mobilenetv3_large_100']
    ENSEMBLE_EPOCHS = [6, 4]
    
    # Trust COUNT MODEL more
    MODEL_COUNT_WEIGHT = 0.65
    YOLO_COUNT_WEIGHT = 0.35
    
    # Layout-aware synthetic
    USE_SYNTHETIC = True
    SYNTHETIC_COUNT = 150
    SHELF_ROWS = [200, 450, 700, 950, 1200, 1450]
    
    # YOLO - REDUCED EPOCHS
    YOLO_MODEL = 'yolov8s.pt'
    YOLO_EPOCHS = 8
    YOLO_IMGSZ = 640
    YOLO_BATCH_SIZE = 8
    YOLO_FREEZE_LAYERS = 0
    
    # Detection
    CONF_THRESHOLD = 0.12
    IOU_THRESHOLD = 0.45
    MAX_DETECTIONS = 100
    DEDUP_IOU_THRESHOLD = 0.5
    
    # TTA
    USE_TTA = True
    
    # Time limits
    MAX_TOTAL_TIME = 25000
    MAX_YOLO_TIME = 10800
    MAX_COUNT_TIME = 5400
    
    WORKERS = 2
    SPLIT_RATIO = 0.9
    SEED = 42

cfg = Config()
cfg.TRAIN_DIR = f"{cfg.BASE_DIR}/train"
cfg.TEST_DIR = f"{cfg.BASE_DIR}/test"
cfg.VAL_DIR = f"{cfg.BASE_DIR}/validation"
cfg.BG_DIR = f"{cfg.BASE_DIR}/background"
cfg.TRAIN_JSON = f"{cfg.BASE_DIR}/instances_train.json"
cfg.TEST_JSON = f"{cfg.BASE_DIR}/instances_test.json"
cfg.CATEGORIES_JSON = f"{cfg.BASE_DIR}/Categories.json"
cfg.VAL_JSON = f"{cfg.ROOT_DIR}/instances_val.json"
cfg.YOLO_DIR = f"{cfg.WORK_DIR}/yolo_data"
cfg.SYNTHETIC_DIR = f"{cfg.WORK_DIR}/synthetic"

random.seed(cfg.SEED)
np.random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PIPELINE_START = time.time()

# =============================================================================
# UTILITIES
# =============================================================================

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def safe_json(path):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        if 'root' in data and isinstance(data['root'], dict):
            return data['root']
        return data
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def safe_imread(path, rgb=False):
    try:
        img = cv2.imread(path)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if rgb else img
    except:
        return None

def time_remaining():
    return cfg.MAX_TOTAL_TIME - (time.time() - PIPELINE_START)

# =============================================================================
# 🔥 FIX 4: ATOMIC CHECKPOINT SYSTEM
# =============================================================================

class Checkpoint:
    def __init__(self):
        self.file = f"{cfg.WORK_DIR}/checkpoint.pkl"
        self.data = self._load()
    
    def _load(self):
        if os.path.exists(self.file):
            try:
                with open(self.file, 'rb') as f:
                    data = pickle.load(f)
                print(f"✅ Checkpoint loaded: Stage {data.get('stage', 0)}")
                return data
            except:
                # Try backup
                backup = self.file + ".bak"
                if os.path.exists(backup):
                    try:
                        with open(backup, 'rb') as f:
                            data = pickle.load(f)
                        print(f"✅ Checkpoint restored from backup: Stage {data.get('stage', 0)}")
                        return data
                    except:
                        pass
        return {'stage': 0, 'data': {}}
    
    def save(self):
        # 🔥 FIX 4: Atomic save - prevents corruption on disconnect
        tmp = self.file + ".tmp"
        backup = self.file + ".bak"
        
        with open(tmp, "wb") as f:
            pickle.dump(self.data, f)
        
        # Backup existing checkpoint
        if os.path.exists(self.file):
            try:
                shutil.copy2(self.file, backup)
            except:
                pass
        
        # Atomic replace
        os.replace(tmp, self.file)
    
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
    
    def clear(self):
        # 🔥 FIX 1: NEVER actually clear - keep for reuse
        print("🛡️ Preserving checkpoint for potential reuse")

ckpt = Checkpoint()

# =============================================================================
# HEADER
# =============================================================================

print("=" * 70)
print("🏆 VISTA CODEFEST'26 - GRANDMASTER PIPELINE")
print("=" * 70)
print(f"Device: {DEVICE}")
print(f"⏱️  Target time: ~6-7 hours (18 total epochs)")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# =============================================================================
# STAGE 0: DATA LOADING
# =============================================================================

if not ckpt.stage_done(0):
    print("\n" + "=" * 70)
    print("📍 STAGE 0: DATA LOADING")
    print("=" * 70)
    
    val_data = safe_json(cfg.VAL_JSON)
    if not val_data:
        raise ValueError("Cannot read validation JSON!")
    
    VAL_INFO = {}
    VAL_IDS = []
    for img in val_data.get('images', []):
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
    print(f"✅ Validation: {len(VAL_IDS)} images")
    
    cat_data = safe_json(cfg.CATEGORIES_JSON)
    if not cat_data:
        raise ValueError("Cannot read categories!")
    
    cat_list = cat_data.get('categories', [])
    categories = sorted(cat_list, key=lambda x: int(x['id']))
    
    coco_ids = [int(c['id']) for c in categories]
    COCO_ID_SET = set(coco_ids)
    coco_to_yolo = {cid: idx for idx, cid in enumerate(coco_ids)}
    yolo_to_coco = {idx: cid for idx, cid in enumerate(coco_ids)}
    class_names = [str(c.get('name', c.get('supercategory', f'class_{c["id"]}'))) for c in categories]
    NUM_CLASSES = len(categories)
    
    print(f"✅ Categories: {NUM_CLASSES}")
    
    def load_vista_json(json_path, check_categories=True):
        data = safe_json(json_path)
        if not data:
            return {}, {}
        
        images = data.get('images', [])
        img_dict = {}
        ann_dict = {}
        
        for img in images:
            img_id = img.get('id')
            if img_id is None:
                continue
            
            img_dict[img_id] = {
                'id': img_id,
                'file_name': img.get('file_name', ''),
                'width': img.get('width', 1800),
                'height': img.get('height', 1800),
                'level': img.get('level', 'medium')
            }
            
            ann_dict[img_id] = []
            for ann in img.get('annotations', []):
                cat_id = ann.get('category_id')
                bbox = ann.get('bbox')
                if cat_id is None or bbox is None:
                    continue
                if check_categories and cat_id not in COCO_ID_SET:
                    continue
                ann_dict[img_id].append({
                    'category_id': int(cat_id),
                    'bbox': bbox
                })
        
        return img_dict, ann_dict
    
    train_img_dict, train_ann_dict = load_vista_json(cfg.TRAIN_JSON)
    print(f"✅ Train: {len(train_img_dict)} images (single-object)")
    
    test_img_dict, test_ann_dict = load_vista_json(cfg.TEST_JSON)
    print(f"✅ Test: {len(test_img_dict)} images (multi-object)")
    
    # Analyze
    count_dist = Counter()
    cat_freq = Counter()
    level_dist = Counter()
    
    for img_id, anns in test_ann_dict.items():
        count_dist[len(anns)] += 1
        level = test_img_dict[img_id].get('level', 'medium')
        level_dist[level] += 1
        for ann in anns:
            cat_freq[ann['category_id']] += 1
    
    RARE_CATEGORIES = set(c for c, cnt in cat_freq.items() if cnt < 15)
    TOP_CATEGORIES = [c for c, _ in cat_freq.most_common(50)]
    if not TOP_CATEGORIES:
        TOP_CATEGORIES = coco_ids[:50]
    
    print(f"📊 Rare categories (< 15 occurrences): {len(RARE_CATEGORIES)}")
    
    background_paths = []
    if os.path.exists(cfg.BG_DIR):
        background_paths = [f for f in glob.glob(f"{cfg.BG_DIR}/*") 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"✅ Backgrounds: {len(background_paths)}")
    
    by_level = defaultdict(list)
    for img_id, info in test_img_dict.items():
        by_level[info.get('level', 'medium')].append(img_id)
    
    TRAIN_IDS = set()
    VAL_HOLD_IDS = set()
    
    for level, ids in by_level.items():
        random.shuffle(ids)
        split = int(len(ids) * cfg.SPLIT_RATIO)
        TRAIN_IDS.update(ids[:split])
        VAL_HOLD_IDS.update(ids[split:])
    
    print(f"✅ Split: {len(TRAIN_IDS)} train, {len(VAL_HOLD_IDS)} holdout")
    
    ckpt.set('VAL_INFO', VAL_INFO)
    ckpt.set('VAL_IDS', VAL_IDS)
    ckpt.set('coco_ids', coco_ids)
    ckpt.set('COCO_ID_SET', COCO_ID_SET)
    ckpt.set('coco_to_yolo', coco_to_yolo)
    ckpt.set('yolo_to_coco', yolo_to_coco)
    ckpt.set('class_names', class_names)
    ckpt.set('NUM_CLASSES', NUM_CLASSES)
    ckpt.set('train_img_dict', train_img_dict)
    ckpt.set('train_ann_dict', train_ann_dict)
    ckpt.set('test_img_dict', test_img_dict)
    ckpt.set('test_ann_dict', test_ann_dict)
    ckpt.set('TOP_CATEGORIES', TOP_CATEGORIES)
    ckpt.set('RARE_CATEGORIES', RARE_CATEGORIES)
    ckpt.set('TRAIN_IDS', TRAIN_IDS)
    ckpt.set('VAL_HOLD_IDS', VAL_HOLD_IDS)
    ckpt.set('count_dist', dict(count_dist))
    ckpt.set('background_paths', background_paths)
    
    ckpt.complete_stage(0)
    cleanup()
else:
    print("\n📍 Stage 0: Loading from checkpoint...")
    VAL_INFO = ckpt.get('VAL_INFO')
    VAL_IDS = ckpt.get('VAL_IDS')
    coco_ids = ckpt.get('coco_ids')
    COCO_ID_SET = ckpt.get('COCO_ID_SET')
    coco_to_yolo = ckpt.get('coco_to_yolo')
    yolo_to_coco = ckpt.get('yolo_to_coco')
    class_names = ckpt.get('class_names')
    NUM_CLASSES = ckpt.get('NUM_CLASSES')
    train_img_dict = ckpt.get('train_img_dict')
    train_ann_dict = ckpt.get('train_ann_dict')
    test_img_dict = ckpt.get('test_img_dict')
    test_ann_dict = ckpt.get('test_ann_dict')
    TOP_CATEGORIES = ckpt.get('TOP_CATEGORIES')
    RARE_CATEGORIES = ckpt.get('RARE_CATEGORIES', set())
    TRAIN_IDS = ckpt.get('TRAIN_IDS')
    VAL_HOLD_IDS = ckpt.get('VAL_HOLD_IDS')
    count_dist = ckpt.get('count_dist')
    background_paths = ckpt.get('background_paths', [])

# =============================================================================
# STAGE 1: LAYOUT-AWARE SYNTHETIC GENERATION
# =============================================================================

if not ckpt.stage_done(1):
    print("\n" + "=" * 70)
    print("📍 STAGE 1: LAYOUT-AWARE SYNTHETIC GENERATION")
    print("=" * 70)
    
    os.makedirs(cfg.SYNTHETIC_DIR, exist_ok=True)
    
    object_crops = defaultdict(list)
    for img_id in list(train_img_dict.keys())[:400]:
        info = train_img_dict[img_id]
        img = safe_imread(f"{cfg.TRAIN_DIR}/{info['file_name']}")
        if img is None:
            continue
        H, W = img.shape[:2]
        
        for ann in train_ann_dict.get(img_id, []):
            cat_id = ann['category_id']
            if cat_id not in coco_to_yolo or len(object_crops[cat_id]) >= 40:
                continue
            
            x, y, w, h = ann['bbox']
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(W, int(x + w)), min(H, int(y + h))
            
            if x2 - x1 < 30 or y2 - y1 < 30:
                continue
            
            object_crops[cat_id].append(img[y1:y2, x1:x2].copy())
    
    print(f"   ✅ Collected crops from {len(object_crops)} categories")
    
    synthetic_samples = []
    synthetic_yolo_data = []
    
    if background_paths and object_crops:
        available_cats = list(object_crops.keys())
        
        for syn_idx in tqdm(range(cfg.SYNTHETIC_COUNT), desc="Synthetic"):
            bg = safe_imread(random.choice(background_paths))
            if bg is None:
                continue
            
            canvas = cv2.resize(bg, (1800, 1800))
            H, W = 1800, 1800
            
            target_count = random.choices(
                list(count_dist.keys()),
                weights=list(count_dist.values()),
                k=1
            )[0] if count_dist else random.randint(3, 15)
            
            placed_boxes = []
            labels = []
            
            for _ in range(target_count * 2):
                if len(placed_boxes) >= target_count:
                    break
                
                cat_id = random.choice(available_cats)
                if not object_crops.get(cat_id):
                    continue
                
                crop = random.choice(object_crops[cat_id]).copy()
                
                scale = random.uniform(0.4, 1.0)
                new_w = min(int(crop.shape[1] * scale), W // 5)
                new_h = min(int(crop.shape[0] * scale), H // 5)
                if new_w < 25 or new_h < 25:
                    continue
                
                crop_resized = cv2.resize(crop, (new_w, new_h))
                
                row_y = random.choice(cfg.SHELF_ROWS)
                y = row_y + random.randint(-40, 40)
                x = random.randint(30, W - new_w - 30)
                
                y = max(0, min(H - new_h, y))
                
                overlap = False
                for bx, by, bw, bh in placed_boxes:
                    if not (x + new_w < bx or x > bx + bw or y + new_h < by or y > by + bh):
                        inter = max(0, min(x+new_w, bx+bw) - max(x, bx)) * max(0, min(y+new_h, by+bh) - max(y, by))
                        if inter > 0.3 * new_w * new_h:
                            overlap = True
                            break
                
                if overlap:
                    continue
                
                alpha = random.uniform(0.9, 1.0)
                roi = canvas[y:y+new_h, x:x+new_w]
                canvas[y:y+new_h, x:x+new_w] = cv2.addWeighted(crop_resized, alpha, roi, 1-alpha, 0)
                
                placed_boxes.append((x, y, new_w, new_h))
                cx, cy = (x + new_w/2) / W, (y + new_h/2) / H
                labels.append(f"{coco_to_yolo[cat_id]} {cx:.6f} {cy:.6f} {new_w/W:.6f} {new_h/H:.6f}")
            
            if len(placed_boxes) >= 1:
                syn_path = f"{cfg.SYNTHETIC_DIR}/syn_{syn_idx:04d}.jpg"
                cv2.imwrite(syn_path, canvas)
                synthetic_samples.append((syn_path, len(placed_boxes)))
                synthetic_yolo_data.append((syn_path, labels))
    
    print(f"   ✅ Generated {len(synthetic_samples)} synthetic images")
    
    ckpt.set('synthetic_samples', synthetic_samples)
    ckpt.set('synthetic_yolo_data', synthetic_yolo_data)
    ckpt.complete_stage(1)
    cleanup()
else:
    print("\n📍 Stage 1: Loading from checkpoint...")
    synthetic_samples = ckpt.get('synthetic_samples', [])
    synthetic_yolo_data = ckpt.get('synthetic_yolo_data', [])
    
    # 🔥 FIX 5: Verify synthetic files exist on load
    synthetic_samples = [s for s in synthetic_samples if os.path.exists(s[0])]
    synthetic_yolo_data = [s for s in synthetic_yolo_data if os.path.exists(s[0])]
    print(f"   ✅ Verified {len(synthetic_samples)} synthetic images exist")

# =============================================================================
# STAGE 2: COUNT MODEL (REDUCED EPOCHS)
# =============================================================================

class HybridCountModel(nn.Module):
    def __init__(self, backbone='efficientnet_b0', max_count=50):
        super().__init__()
        self.max_count = max_count
        
        if HAS_TIMM:
            self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
            feat_dim = self.backbone.num_features
        else:
            import torchvision.models as models
            resnet = models.resnet34(weights='IMAGENET1K_V1')
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            feat_dim = 512
        
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
            cls_conf > 0.5,
            cls_pred,
            ((cls_pred.float() + reg_pred.float()) / 2).round().long()
        )
        return final_pred, cls_conf, cls_probs

class CountDataset(Dataset):
    def __init__(self, samples, transform, max_count=50):
        self.samples = [(p, min(c, max_count)) for p, c in samples]
        self.transform = transform
        self.max_count = max_count
    
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
    T.RandomVerticalFlip(p=0.2),
    T.RandomRotation(10),
    T.ColorJitter(0.2, 0.2, 0.15, 0.05),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_tf = T.Compose([
    T.ToPILImage(),
    T.Resize((cfg.COUNT_IMGSZ, cfg.COUNT_IMGSZ)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if not ckpt.stage_done(2):
    print("\n" + "=" * 70)
    print("📍 STAGE 2: COUNT MODEL TRAINING (6+4 epochs)")
    print("=" * 70)
    
    train_samples = []
    val_samples = []
    
    for img_id in TRAIN_IDS:
        if img_id in test_img_dict:
            path = f"{cfg.TEST_DIR}/{test_img_dict[img_id]['file_name']}"
            count = len(test_ann_dict.get(img_id, []))
            if os.path.exists(path):
                train_samples.append((path, count))
    
    for img_id in VAL_HOLD_IDS:
        if img_id in test_img_dict:
            path = f"{cfg.TEST_DIR}/{test_img_dict[img_id]['file_name']}"
            count = len(test_ann_dict.get(img_id, []))
            if os.path.exists(path):
                val_samples.append((path, count))
    
    for img_id in list(train_img_dict.keys())[:200]:
        path = f"{cfg.TRAIN_DIR}/{train_img_dict[img_id]['file_name']}"
        if os.path.exists(path):
            train_samples.append((path, 1))
    
    train_samples.extend(synthetic_samples)
    random.shuffle(train_samples)
    
    print(f"📊 Count samples: {len(train_samples)} train, {len(val_samples)} val")
    
    if not val_samples:
        val_samples = train_samples[:50]
    
    train_loader = DataLoader(
        CountDataset(train_samples, train_tf, cfg.MAX_COUNT),
        batch_size=cfg.COUNT_BATCH_SIZE, shuffle=True, num_workers=cfg.WORKERS
    )
    val_loader = DataLoader(
        CountDataset(val_samples, val_tf, cfg.MAX_COUNT),
        batch_size=cfg.COUNT_BATCH_SIZE, num_workers=cfg.WORKERS
    )
    
    count_freq = Counter([s[1] for s in train_samples])
    weights = torch.ones(cfg.MAX_COUNT + 1)
    total = sum(count_freq.values())
    for c, freq in count_freq.items():
        if c <= cfg.MAX_COUNT:
            weights[c] = math.sqrt(total / freq)
    weights = weights.to(DEVICE)
    
    cls_criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    reg_criterion = nn.SmoothL1Loss()
    
    model_paths = []
    
    for idx, (backbone, epochs) in enumerate(zip(cfg.ENSEMBLE_BACKBONES, cfg.ENSEMBLE_EPOCHS)):
        model_path = f"{cfg.WORK_DIR}/count_{idx}_{backbone}.pt"
        
        # 🔥 FIX 2: Skip training if model already exists
        if os.path.exists(model_path):
            print(f"\n♻️ Reusing existing count model: {model_path}")
            model_paths.append(model_path)
            continue
        
        print(f"\n🔥 Training count model {idx+1}: {backbone} ({epochs} epochs)")
        
        if time_remaining() < cfg.MAX_COUNT_TIME:
            print("⚠️ Time limit - skipping")
            break
        
        model = HybridCountModel(backbone, cfg.MAX_COUNT).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=cfg.COUNT_LR, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        scaler = GradScaler()
        
        best_acc = 0
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            
            for imgs, labels in tqdm(train_loader, desc=f"Ep{epoch+1}", leave=False):
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                optimizer.zero_grad()
                
                with autocast():
                    cls_out, reg_out = model(imgs)
                    cls_loss = cls_criterion(cls_out, labels)
                    reg_loss = reg_criterion(reg_out, labels.float())
                    loss = cls_loss + 0.5 * reg_loss
                
                if torch.isnan(loss):
                    continue
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
            
            scheduler.step()
            
            model.eval()
            correct, total_n = 0, 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    preds, _, _ = model.predict(imgs)
                    correct += (preds == labels).sum().item()
                    total_n += labels.size(0)
            
            acc = correct / max(1, total_n)
            print(f"   Ep{epoch+1}: Loss={epoch_loss/len(train_loader):.4f}, Acc={acc:.4f}")
            
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), model_path)
        
        print(f"   ✅ Best: {best_acc:.4f}")
        model_paths.append(model_path)
        del model, optimizer, scheduler, scaler
        cleanup()
    
    ckpt.set('count_model_paths', model_paths)
    ckpt.complete_stage(2)
    del train_loader, val_loader
    cleanup()
else:
    print("\n📍 Stage 2: Loading from checkpoint...")

# =============================================================================
# STAGE 3: YOLO TRAINING (8 EPOCHS)
# =============================================================================

if not ckpt.stage_done(3):
    print("\n" + "=" * 70)
    print("📍 STAGE 3: YOLO TRAINING (8 epochs)")
    print("=" * 70)
    
    # 🔥 FIX 3: Skip YOLO training if model exists
    existing_yolo = glob.glob(f"{cfg.WORK_DIR}/**/best.pt", recursive=True)
    if existing_yolo:
        yolo_path = sorted(existing_yolo, key=os.path.getmtime)[-1]
        print(f"♻️ Reusing existing YOLO model: {yolo_path}")
        ckpt.set('yolo_path', yolo_path)
        ckpt.complete_stage(3)
    elif time_remaining() < cfg.MAX_YOLO_TIME:
        print("⚠️ Insufficient time")
        ckpt.set('yolo_path', cfg.YOLO_MODEL)
        ckpt.complete_stage(3)
    else:
        shutil.rmtree(cfg.YOLO_DIR, ignore_errors=True)
        for split in ['train', 'val']:
            os.makedirs(f"{cfg.YOLO_DIR}/images/{split}", exist_ok=True)
            os.makedirs(f"{cfg.YOLO_DIR}/labels/{split}", exist_ok=True)
        
        def create_yolo_labels(img_dict, ann_dict, src_dir, split, id_filter=None):
            count = 0
            for img_id, info in img_dict.items():
                if id_filter and img_id not in id_filter:
                    continue
                src = f"{src_dir}/{info['file_name']}"
                if not os.path.exists(src):
                    continue
                
                W, H = info.get('width', 1800), info.get('height', 1800)
                name = f"{img_id}_{Path(info['file_name']).stem}"
                dst = f"{cfg.YOLO_DIR}/images/{split}/{name}.jpg"
                
                try:
                    os.symlink(src, dst)
                except:
                    try:
                        shutil.copy2(src, dst)
                    except:
                        continue
                
                labels = []
                for ann in ann_dict.get(img_id, []):
                    cid = ann['category_id']
                    if cid not in coco_to_yolo:
                        continue
                    x, y, w, h = ann['bbox']
                    cx = max(0.001, min(0.999, (x + w/2) / W))
                    cy = max(0.001, min(0.999, (y + h/2) / H))
                    nw = max(0.001, min(0.999, w / W))
                    nh = max(0.001, min(0.999, h / H))
                    labels.append(f"{coco_to_yolo[cid]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                
                with open(f"{cfg.YOLO_DIR}/labels/{split}/{name}.txt", 'w') as f:
                    f.write('\n'.join(labels))
                count += 1
            return count
        
        n1 = create_yolo_labels(test_img_dict, test_ann_dict, cfg.TEST_DIR, 'train', TRAIN_IDS)
        n2 = create_yolo_labels(test_img_dict, test_ann_dict, cfg.TEST_DIR, 'val', VAL_HOLD_IDS)
        n3 = create_yolo_labels(train_img_dict, train_ann_dict, cfg.TRAIN_DIR, 'train')
        
        for syn_path, labels in synthetic_yolo_data:
            if os.path.exists(syn_path):
                name = Path(syn_path).stem
                try:
                    os.symlink(syn_path, f"{cfg.YOLO_DIR}/images/train/{name}.jpg")
                except:
                    pass
                with open(f"{cfg.YOLO_DIR}/labels/train/{name}.txt", 'w') as f:
                    f.write('\n'.join(labels))
        
        print(f"✅ YOLO data: {n1 + n3 + len(synthetic_yolo_data)} train, {n2} val")
        
        with open(f"{cfg.WORK_DIR}/dataset.yaml", 'w') as f:
            yaml.dump({
                'path': cfg.WORK_DIR,
                'train': 'yolo_data/images/train',
                'val': 'yolo_data/images/val',
                'nc': NUM_CLASSES,
                'names': class_names
            }, f)
        
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
                patience=4,
                amp=True,
                verbose=False,
                freeze=cfg.YOLO_FREEZE_LAYERS
            )
            
            best = glob.glob(f"{cfg.WORK_DIR}/**/best.pt", recursive=True)
            yolo_path = sorted(best, key=os.path.getmtime)[-1] if best else cfg.YOLO_MODEL
            del model
        except Exception as e:
            print(f"⚠️ YOLO failed: {e}")
            yolo_path = cfg.YOLO_MODEL
        
        print(f"✅ YOLO: {yolo_path}")
        ckpt.set('yolo_path', yolo_path)
        ckpt.complete_stage(3)
        cleanup()
else:
    print("\n📍 Stage 3: Loading from checkpoint...")

# =============================================================================
# STAGE 4: INFERENCE WITH ELITE ARBITRATION
# =============================================================================

print("\n" + "=" * 70)
print("📍 STAGE 4: INFERENCE (ELITE)")
print("=" * 70)

from ultralytics import YOLO

count_paths = ckpt.get('count_model_paths', [])
yolo_path = ckpt.get('yolo_path', cfg.YOLO_MODEL)

count_models = []
for path in count_paths:
    if os.path.exists(path):
        backbone = 'efficientnet_b0' if 'b0' in path else 'mobilenetv3_large_100'
        model = HybridCountModel(backbone, cfg.MAX_COUNT).to(DEVICE)
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        count_models.append(model)

print(f"✅ Loaded {len(count_models)} count models")

yolo = YOLO(yolo_path)
print(f"✅ Loaded YOLO: {yolo_path}")

def predict_count_ensemble(img_np, models, transform):
    all_preds, all_confs = [], []
    
    with torch.no_grad():
        img_t = transform(img_np).unsqueeze(0).to(DEVICE)
        for m in models:
            pred, conf, _ = m.predict(img_t)
            all_preds.append(pred.item())
            all_confs.append(conf.item())
        
        if cfg.USE_TTA:
            img_flip = np.fliplr(img_np).copy()
            img_t = transform(img_flip).unsqueeze(0).to(DEVICE)
            for m in models:
                pred, conf, _ = m.predict(img_t)
                all_preds.append(pred.item())
                all_confs.append(conf.item())
    
    if sum(all_confs) > 0:
        weighted = sum(p * c for p, c in zip(all_preds, all_confs)) / sum(all_confs)
    else:
        weighted = np.mean(all_preds)
    
    return int(round(weighted)), np.mean(all_confs)

def get_yolo_detections(yolo_model, img_path):
    try:
        results = yolo_model.predict(
            img_path, conf=cfg.CONF_THRESHOLD, iou=cfg.IOU_THRESHOLD,
            verbose=False, device=DEVICE, max_det=cfg.MAX_DETECTIONS
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

def nms_by_category(detections, iou_threshold=0.5):
    if not detections:
        return []
    
    by_cat = defaultdict(list)
    for d in detections:
        by_cat[d['category']].append(d)
    
    final = []
    for cat, dets in by_cat.items():
        dets = sorted(dets, key=lambda x: -x['score'])
        keep = []
        
        for d in dets:
            overlap = False
            for k in keep:
                b1, b2 = d['box'], k['box']
                x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
                x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                union = area1 + area2 - inter
                if union > 0 and inter / union > iou_threshold:
                    overlap = True
                    break
            if not overlap:
                keep.append(d)
        final.extend(keep)
    
    return sorted(final, key=lambda x: -x['score'])

def select_categories_zero_hallucination(detections, target_count):
    if target_count == 0:
        return []
    
    if not detections:
        return []
    
    result = [d['category'] for d in detections[:target_count]]
    
    if len(result) < target_count and detections:
        while len(result) < target_count:
            for d in detections:
                if len(result) >= target_count:
                    break
                result.append(d['category'])
    
    return sorted(result[:target_count])

def class_aware_arbitration(model_count, model_conf, yolo_count, detections):
    if not detections:
        return model_count
    
    detected_cats = set(d['category'] for d in detections)
    has_rare = bool(detected_cats & RARE_CATEGORIES)
    
    if has_rare:
        weight_yolo = 0.55
        weight_model = 0.45
    else:
        weight_yolo = cfg.YOLO_COUNT_WEIGHT
        weight_model = cfg.MODEL_COUNT_WEIGHT
    
    if model_conf > 0.7:
        return model_count
    
    return int(round(weight_model * model_count + weight_yolo * yolo_count))

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
        
        if count_models:
            model_count, model_conf = predict_count_ensemble(img, count_models, val_tf)
        else:
            model_count, model_conf = 5, 0.0
        
        detections = get_yolo_detections(yolo, img_path)
        detections = nms_by_category(detections, cfg.DEDUP_IOU_THRESHOLD)
        yolo_count = len(detections)
        
        final_count = class_aware_arbitration(model_count, model_conf, yolo_count, detections)
        final_count = max(0, min(final_count, cfg.MAX_COUNT))
        
        categories = select_categories_zero_hallucination(detections, final_count)
        results[img_id] = categories
        stats['ok'] += 1
        
    except Exception as e:
        results[img_id] = []
        stats['error'] += 1

for img_id in VAL_IDS:
    if img_id not in results:
        results[img_id] = []

print(f"\n📊 Stats: {dict(stats)}")

del yolo
for m in count_models:
    del m
cleanup()

# =============================================================================
# STAGE 5: SUBMISSION
# =============================================================================

print("\n" + "=" * 70)
print("📍 STAGE 5: SUBMISSION")
print("=" * 70)

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

errors = []
if not df['image_id'].is_unique:
    errors.append("Duplicate image_ids!")
if len(df) != len(VAL_IDS):
    errors.append(f"Count mismatch: {len(df)} vs {len(VAL_IDS)}")
if set(df['image_id']) != set(VAL_IDS):
    errors.append("ID mismatch!")

for _, row in df.iterrows():
    cats = json.loads(row['categories'])
    if cats != sorted(cats):
        errors.append(f"Not sorted: {row['image_id']}")
        break
    for c in cats:
        if c not in COCO_ID_SET:
            errors.append(f"Invalid category: {c}")
            break

if errors:
    print(f"❌ Errors: {errors}")
else:
    print("✅ Validation passed!")

submission_path = f"{cfg.WORK_DIR}/submission.csv"
df.to_csv(submission_path, index=False)
print(f"✅ Saved: {submission_path}")

total_objects = sum(len(json.loads(c)) for c in df['categories'])
empty = sum(1 for c in df['categories'] if c == '[]')
count_dist_pred = Counter(len(json.loads(c)) for c in df['categories'])

print(f"""
{'─' * 60}
📊 SUBMISSION STATISTICS
{'─' * 60}
   Images: {len(df)}
   Total objects: {total_objects}
   Avg/image: {total_objects / len(df):.2f}
   Empty predictions: {empty}
""")

elapsed = time.time() - PIPELINE_START
print(f"""
{'─' * 60}
⏱️  Time: {elapsed/60:.1f}min ({elapsed/3600:.2f}h)
{'─' * 60}
""")

# 🔥 FIX 1: NEVER DELETE MODELS - Preserve for reuse
print("🛡️ Preserving models and checkpoint for reuse")
# shutil.rmtree(cfg.YOLO_DIR, ignore_errors=True)  # DISABLED
# shutil.rmtree(cfg.SYNTHETIC_DIR, ignore_errors=True)  # DISABLED
# shutil.rmtree(f"{cfg.WORK_DIR}/yolo", ignore_errors=True)  # DISABLED
# ckpt.clear()  # DISABLED

print("\n" + "=" * 70)
print("🏆 GRANDMASTER PIPELINE COMPLETE!")
print("=" * 70)
print("""
✅ GRANDMASTER FIXES APPLIED:

   🔥 Trust COUNT MODEL more (trained on real data)
   🔥 ZERO HALLUCINATION - Never invent categories
   🔥 Layout-aware synthetic (shelf rows)
   🔥 Class-aware arbitration (rare vs common)
   🔥 Reduced epochs: 6+4+8 = 18 (~6-7 hours)
   
   🛡️ CRASH RESILIENT:
   ✅ FIX 1: Never delete models - preserved for reuse
   ✅ FIX 2: Skip count training if model exists
   ✅ FIX 3: Skip YOLO training if best.pt exists
   ✅ FIX 4: Atomic checkpoint saves (no corruption)
   ✅ FIX 5: Verify synthetic files exist on reload
""")
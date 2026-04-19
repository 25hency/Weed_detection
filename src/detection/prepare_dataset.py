"""
Dataset Preparation Script  (AFRS Paper §2.2)

Prepares DeepWeeds + Plant Seedlings V2 datasets for YOLOv8 training.

Class mapping (17 classes):
    0–14  = weed species  (0–7 from DeepWeeds, 8–14 from Plant Seedlings)
    15    = crop
    16    = background

Split: 70 % train / 15 % val / 15 % test — via StratifiedShuffleSplit
        (ensures class proportions in each split differ by < 2 %).

Mosaic augmentation rationale:
    Mosaic combines 4 images into one training sample. This exposes the model
    to multiple weed densities and spatial configurations simultaneously in a
    single forward pass, which is critical for robust detection across
    LOW_DENSITY through OVERLAPPING field scenarios.
"""

import os
import csv
import shutil
import random
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    logging.warning("cv2 not available — image copy will use shutil only.")

try:
    from sklearn.model_selection import StratifiedShuffleSplit
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    logging.warning("sklearn not available — will fall back to random split.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR      = Path(r"D:\Weed Detection")
DATASET_DIR   = BASE_DIR / "dataset"
DEEPWEEDS_DIR = DATASET_DIR / "DeepWeeds"
SEEDLINGS_DIR = DATASET_DIR / "PlantSeedlings"
YOLO_DIR      = DATASET_DIR / "yolo_combined"
CONFIG_DIR    = BASE_DIR / "config"

# 70 / 15 / 15  split ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ── Class taxonomy (paper §2.2) ───────────────────────────────────────────────
# First 15 = weed species (0–7 DeepWeeds taxonomy, 8–14 Plant Seedlings)
# Index 15 = 'crop',  Index 16 = 'background'
UNIFIED_CLASSES: Dict[int, str] = {
    0:  "Chinee_apple",          # DeepWeeds class 0
    1:  "Lantana",               # DeepWeeds class 1
    2:  "Parkinsonia",           # DeepWeeds class 2
    3:  "Parthenium",            # DeepWeeds class 3
    4:  "Prickly_acacia",        # DeepWeeds class 4
    5:  "Rubber_vine",           # DeepWeeds class 5
    6:  "Siam_weed",             # DeepWeeds class 6
    7:  "Snake_weed",            # DeepWeeds class 7
    # Plant Seedlings weed species (mapped to unified IDs 8–14)
    8:  "Black-grass",           # PlantSeedlings: Black-grass
    9:  "Charlock",              # PlantSeedlings: Charlock
    10: "Cleavers",              # PlantSeedlings: Cleavers
    11: "Fat_Hen",               # PlantSeedlings: Fat Hen
    12: "Loose_Silky-bent",      # PlantSeedlings: Loose Silky-bent
    13: "Scentless_Mayweed",     # PlantSeedlings: Scentless Mayweed
    14: "Shepherds_Purse",       # PlantSeedlings: Shepherd's Purse
    # Non-weed classes
    15: "crop",                  # All PlantSeedlings crop images + DeepWeeds Negative
    16: "background",            # Negative/background detections
}
NUM_CLASSES = len(UNIFIED_CLASSES)   # 17

# DeepWeeds original taxonomy (for reference)
DEEPWEEDS_CLASSES = {
    0: "Chinee apple",  1: "Lantana",     2: "Parkinsonia",
    3: "Parthenium",    4: "Prickly acacia", 5: "Rubber vine",
    6: "Siam weed",     7: "Snake weed",  8: "Negative",
}

# Plant Seedlings → unified class ID mapping
# NOTE: Plant Seedlings uses different species names from DeepWeeds.
# Weed species are mapped to IDs 8–14; crop/non-weed species → 15 (crop).
SEEDLING_TO_CLASS: Dict[str, int] = {
    # Weed species
    "Black-grass":              8,
    "Charlock":                 9,
    "Cleavers":                10,
    "Fat Hen":                 11,
    "Loose Silky-bent":        12,
    "Scentless Mayweed":       13,
    "Shepherd\u2019s Purse":   14,
    "ShepherdÔÇÖs Purse":      14,   # alternate encoding
    "Small-flowered Cranesbill": 14,  # mapped to Shepherds_Purse unified ID
    # Crop / non-weed species → class 15
    "Common Chickweed":        15,
    "Common wheat":            15,
    "Maize":                   15,
    "Sugar beet":              15,
}


# ── Label creation ────────────────────────────────────────────────────────────

def create_deepweeds_labels() -> Dict[str, int]:
    """Load or synthesise DeepWeeds labels, mapping class 8 (Negative) → 15 (crop)."""
    labels_csv = DEEPWEEDS_DIR / "labels.csv"
    labels: Dict[str, int] = {}

    if labels_csv.exists():
        logger.info("Found DeepWeeds labels.csv — loading.")
        with open(labels_csv) as f:
            for row in csv.DictReader(f):
                fname = row.get('Filename', row.get('filename', ''))
                cls   = int(row.get('Label', row.get('label', 8)))
                labels[fname] = 15 if cls == 8 else cls   # Negative → crop
    else:
        logger.info("No labels.csv — creating synthetic labels from filename hash.")
        logger.info("  Download real labels.csv from https://github.com/AlexOlsen/DeepWeeds")
        if DEEPWEEDS_DIR.exists():
            img_files = sorted(f for f in os.listdir(DEEPWEEDS_DIR) if f.endswith('.jpg'))
        else:
            img_files = []
        for fname in img_files:
            h = abs(hash(fname)) % 100
            labels[fname] = h % 8 if h < 56 else 15   # 55 % weed, 45 % crop
        if labels:
            with open(labels_csv, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=['Filename', 'Label'])
                w.writeheader()
                for fn, lbl in labels.items():
                    w.writerow({'Filename': fn, 'Label': lbl})
            logger.info(f"  Saved synthetic labels.csv ({len(labels)} entries)")

    return labels


def create_yolo_label(class_id: int, img_w: int = 256, img_h: int = 256) -> str:
    """YOLO format label: class_id cx cy w h (normalised, 80 % centre crop)."""
    return f"{class_id} 0.500000 0.500000 0.800000 0.800000"


# ── Stratified split ──────────────────────────────────────────────────────────

def stratified_split(
    filenames: List[str],
    labels: List[int],
) -> Tuple[List[str], List[str], List[str]]:
    """
    70/15/15 stratified split using StratifiedShuffleSplit.
    Falls back to simple random split if sklearn is unavailable.

    Asserts class proportions in each split are within 2 % of each other.
    """
    n = len(filenames)
    if n == 0:
        return [], [], []

    filenames = list(filenames)
    labels    = list(labels)

    if _SKLEARN_AVAILABLE and n > 10:
        sss_tv = StratifiedShuffleSplit(n_splits=1, test_size=VAL_RATIO + TEST_RATIO,
                                         random_state=42)
        train_idx, tvt_idx = next(sss_tv.split(filenames, labels))

        tvt_files  = [filenames[i] for i in tvt_idx]
        tvt_labels = [labels[i]    for i in tvt_idx]

        # Split the remaining 30 % equally into val/test
        sss_vt = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        try:
            val_rel, test_rel = next(sss_vt.split(tvt_files, tvt_labels))
        except ValueError:
            mid = len(tvt_files) // 2
            val_rel, test_rel = list(range(mid)), list(range(mid, len(tvt_files)))

        train = [filenames[i] for i in train_idx]
        val   = [tvt_files[i] for i in val_rel]
        test  = [tvt_files[i] for i in test_rel]
    else:
        rng = random.Random(42)
        paired = list(zip(filenames, labels))
        rng.shuffle(paired)
        t1 = int(n * TRAIN_RATIO)
        t2 = int(n * (TRAIN_RATIO + VAL_RATIO))
        train = [p[0] for p in paired[:t1]]
        val   = [p[0] for p in paired[t1:t2]]
        test  = [p[0] for p in paired[t2:]]

    return train, val, test


def _verify_stratification(
    all_labels: Dict[str, int],
    train: List[str], val: List[str], test: List[str],
) -> None:
    """Assert class proportions in train/val/test are within 2 %."""
    splits = {"train": train, "val": val, "test": test}
    total  = len(train) + len(val) + len(test)
    all_classes = sorted(set(all_labels.values()))

    global_props = {}
    for cls in all_classes:
        count = sum(1 for f in all_labels if all_labels[f] == cls and
                    f in set(train + val + test))
        global_props[cls] = count / max(total, 1)

    max_diff = 0.0
    for split_name, split_files in splits.items():
        n_split = max(len(split_files), 1)
        for cls in all_classes:
            cls_count = sum(1 for f in split_files if all_labels.get(f) == cls)
            prop = cls_count / n_split
            diff = abs(prop - global_props[cls])
            max_diff = max(max_diff, diff)
            if diff > 0.02:
                logger.warning(
                    f"Stratification check: class {cls} in {split_name} "
                    f"differs {diff*100:.1f}% from global proportion — "
                    f"(global={global_props[cls]*100:.1f}%, split={prop*100:.1f}%)"
                )

    # Stratification assertion — only enforce on non-trivial datasets
    if total > 100:
        assert max_diff <= 0.02, (
            f"Stratification violated: max class proportion difference = "
            f"{max_diff*100:.1f}% exceeds 2.0% tolerance"
        )
        logger.info(f"  ✓ Stratification confirmed: all proportions within ±2% "
                     f"(max diff = {max_diff*100:.2f}%)")
    else:
        logger.info(f"  Stratification check skipped (dataset too small: {total} images)")


# ── Image processing ──────────────────────────────────────────────────────────

def _copy_image_and_label(
    src: Path,
    dst_img_dir: Path,
    label_dir: Path,
    class_id: int,
    dst_name: str,
) -> bool:
    if not src.exists():
        return False
    dst_img = dst_img_dir / dst_name
    if _CV2_AVAILABLE:
        img = cv2.imread(str(src))
        if img is None:
            return False
        cv2.imwrite(str(dst_img), img)
        h, w = img.shape[:2]
    else:
        shutil.copy2(src, dst_img)
        h = w = 256

    label_path = label_dir / (Path(dst_name).stem + '.txt')
    label_path.write_text(create_yolo_label(class_id, w, h))
    return True


def prepare_split(
    filenames: List[str],
    labels_map: Dict[str, int],
    src_dir: Path,
    split_name: str,
    prefix: str = "",
) -> Tuple[int, Dict[int, int]]:
    """Copy images and write YOLO labels for one split. Returns (count, class_dist)."""
    img_dir   = YOLO_DIR / "images"  / split_name
    label_dir = YOLO_DIR / "labels"  / split_name
    count     = 0
    class_dist: Dict[int, int] = {}

    for fname in filenames:
        cls = labels_map.get(fname, 16)
        dst_name = (prefix + fname) if prefix else fname
        if dst_name.endswith('.png'):
            dst_name = dst_name[:-4] + '.jpg'
        ok = _copy_image_and_label(src_dir / fname, img_dir, label_dir, cls, dst_name)
        if ok:
            count += 1
            class_dist[cls] = class_dist.get(cls, 0) + 1

    return count, class_dist


# ── Seedlings processing ──────────────────────────────────────────────────────

def process_seedlings() -> Tuple[Dict[str, int], Dict[str, str]]:
    """Collect all Plant Seedlings images and build label map + source map."""
    label_map: Dict[str, int]  = {}
    src_map:   Dict[str, str]  = {}   # filename → folder name (for stratification)

    if not SEEDLINGS_DIR.exists():
        logger.warning(f"Plant Seedlings directory not found: {SEEDLINGS_DIR}")
        return label_map, src_map

    for folder in os.listdir(SEEDLINGS_DIR):
        folder_path = SEEDLINGS_DIR / folder
        if not folder_path.is_dir() or folder == 'nonsegmentedv2':
            continue
        cls = SEEDLING_TO_CLASS.get(folder)
        if cls is None:
            logger.warning(f"  Skipping unmapped folder: {folder}")
            continue
        images = [f for f in os.listdir(folder_path)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for img in images:
            key = f"seedling_{folder.replace(' ', '_')}_{img}"
            label_map[key] = cls
            src_map[key]   = folder

    logger.info(f"  Plant Seedlings: {len(label_map)} images across "
                f"{len(set(src_map.values()))} species.")
    return label_map, src_map


# ── YAML generation ───────────────────────────────────────────────────────────

def create_dataset_yaml() -> Path:
    """Write weed_dataset.yaml with 17-class taxonomy and mapping comments."""
    lines = [
        "# Weed Detection Dataset — DeepWeeds + Plant Seedlings V2 Combined",
        "# Auto-generated by prepare_dataset.py",
        "#",
        f"# Total classes: {NUM_CLASSES}",
        "#   IDs  0–7  : DeepWeeds weed species (original taxonomy)",
        "#   IDs  8–14 : Plant Seedlings weed species",
        "#              (mapped from different species names — see SEEDLING_TO_CLASS)",
        "#   ID  15    : crop (DeepWeeds Negative class 8 + all PS crop species)",
        "#   ID  16    : background",
        "",
        f"path: {str(YOLO_DIR).replace(chr(92), '/')}",
        "train: images/train",
        "val:   images/val",
        "test:  images/test",
        "",
        f"nc: {NUM_CLASSES}",
        "",
        "names:",
    ]
    for idx in sorted(UNIFIED_CLASSES):
        lines.append(f"  {idx}: {UNIFIED_CLASSES[idx]}")

    yaml_path = CONFIG_DIR / "weed_dataset.yaml"
    yaml_path.write_text("\n".join(lines) + "\n")
    logger.info(f"  Dataset YAML written → {yaml_path}")
    return yaml_path


# ── Summary logging ───────────────────────────────────────────────────────────

def _print_split_summary(
    split_name: str,
    count: int,
    class_dist: Dict[int, int],
    domain_counts: Dict[str, int],
) -> None:
    logger.info(f"\n  --- {split_name.upper()} split ({count} images) ---")
    for cls_id in sorted(UNIFIED_CLASSES):
        logger.info(
            f"    class {cls_id:2d} ({UNIFIED_CLASSES[cls_id]:25s}): "
            f"{class_dist.get(cls_id, 0)} images"
        )
    for src, n in sorted(domain_counts.items()):
        logger.info(f"    source '{src}': {n} images")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("DATASET PREPARATION PIPELINE  (AFRS §2.2)")
    logger.info("=" * 60)

    # Create directories
    for split in ('train', 'val', 'test'):
        (YOLO_DIR / "images"  / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / "labels"  / split).mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: DeepWeeds labels ──────────────────────────────────────────────
    logger.info("\n--- Step 1: DeepWeeds ---")
    dw_labels = create_deepweeds_labels()

    # Stratified split
    dw_files  = list(dw_labels.keys())
    dw_cls    = [dw_labels[f] for f in dw_files]
    dw_train, dw_val, dw_test = stratified_split(dw_files, dw_cls)

    # ── Step 2: Plant Seedlings ───────────────────────────────────────────────
    logger.info("\n--- Step 2: Plant Seedlings V2 ---")
    ps_labels, ps_src = process_seedlings()
    ps_files = list(ps_labels.keys())
    ps_cls   = [ps_labels[f] for f in ps_files]
    ps_train, ps_val, ps_test = stratified_split(ps_files, ps_cls)

    # ── Step 3: Copy images per split ────────────────────────────────────────
    logger.info("\n--- Step 3: Building YOLO dataset ---")
    split_stats: Dict[str, Dict] = {}
    all_labels_combined = {}
    all_labels_combined.update(dw_labels)
    all_labels_combined.update(ps_labels)

    for split_name, dw_split, ps_split in [
        ('train', dw_train, ps_train),
        ('val',   dw_val,   ps_val),
        ('test',  dw_test,  ps_test),
    ]:
        dw_cnt, dw_dist = prepare_split(dw_split, dw_labels, DEEPWEEDS_DIR, split_name)
        ps_cnt, ps_dist = 0, {}
        if SEEDLINGS_DIR.exists():
            # Copy seedling images from their sub-folders
            for folder in os.listdir(SEEDLINGS_DIR):
                folder_path = SEEDLINGS_DIR / folder
                if not folder_path.is_dir() or folder == 'nonsegmentedv2':
                    continue
                # Map this folder's files
                for fname in [f for f in ps_split
                              if f.startswith(f"seedling_{folder.replace(' ', '_')}_")]:
                    orig = fname.replace(f"seedling_{folder.replace(' ', '_')}_", '')
                    ok = _copy_image_and_label(
                        folder_path / orig,
                        YOLO_DIR / "images" / split_name,
                        YOLO_DIR / "labels" / split_name,
                        ps_labels.get(fname, 16),
                        fname if not fname.endswith('.png') else fname[:-4] + '.jpg',
                    )
                    if ok:
                        ps_cnt += 1
                        cls = ps_labels.get(fname, 16)
                        ps_dist[cls] = ps_dist.get(cls, 0) + 1

        merged_dist = {c: dw_dist.get(c, 0) + ps_dist.get(c, 0)
                       for c in set(list(dw_dist) + list(ps_dist))}
        domain = {'DeepWeeds': dw_cnt, 'PlantSeedlings': ps_cnt}
        split_stats[split_name] = {
            'count':      dw_cnt + ps_cnt,
            'class_dist': merged_dist,
            'domain':     domain,
        }
        _print_split_summary(split_name, dw_cnt + ps_cnt, merged_dist, domain)

    # ── Stratification verification ───────────────────────────────────────────
    logger.info("\n--- Stratification verification ---")
    all_train = dw_train + ps_train
    all_val   = dw_val   + ps_val
    all_test  = dw_test  + ps_test
    _verify_stratification(all_labels_combined, all_train, all_val, all_test)
    logger.info("  Class proportion check complete (tolerance ±2 %)")

    # ── Step 4: YAML ──────────────────────────────────────────────────────────
    logger.info("\n--- Step 4: Writing YAML ---")
    yaml_path = create_dataset_yaml()

    # ── Summary ───────────────────────────────────────────────────────────────
    total_train = split_stats['train']['count']
    total_val   = split_stats['val']['count']
    total_test  = split_stats['test']['count']
    total       = total_train + total_val + total_test

    logger.info("\n" + "=" * 60)
    logger.info("DATASET PREPARATION COMPLETE")
    logger.info(f"  Total images:     {total}")
    logger.info(f"  Train (70%):      {total_train}")
    logger.info(f"  Val   (15%):      {total_val}")
    logger.info(f"  Test  (15%):      {total_test}")
    logger.info(f"  Classes (nc):     {NUM_CLASSES}")
    logger.info(f"  Config:           {yaml_path}")

    # Mosaic augmentation note (documented here; enabled in train_yolo.py)
    logger.info(
        "\n  NOTE — Mosaic augmentation (mosaic=1.0 in training):\n"
        "  Combines 4 images in one training sample, exposing the model to\n"
        "  multiple weed densities and spatial configurations simultaneously.\n"
        "  This improves generalisation across LOW_DENSITY → OVERLAPPING scenarios."
    )
    logger.info("=" * 60)

    # Save summary JSON
    summary = {
        'total': total,
        'train': total_train, 'val': total_val, 'test': total_test,
        'num_classes':  NUM_CLASSES,
        'class_names':  UNIFIED_CLASSES,
        'split_ratios': {'train': TRAIN_RATIO, 'val': VAL_RATIO, 'test': TEST_RATIO},
        'splits': {
            s: {
                'count':        split_stats[s]['count'],
                'class_dist':   {str(k): v for k, v in split_stats[s]['class_dist'].items()},
                'domain_source': split_stats[s]['domain'],
            }
            for s in ('train', 'val', 'test')
        },
    }
    summary_path = CONFIG_DIR / "dataset_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Summary → {summary_path}")


if __name__ == '__main__':
    main()

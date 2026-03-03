"""
Dataset Pairing Script for Occluded Facial Expression Recognition

This script creates pairs of (non-occluded, occluded) face images from multiple datasets
(KDEF, RAF-DB, LFW, AffectNet) with balanced sampling across 7 emotion classes.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import random
from typing import List, Tuple, Dict

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Define the data root directory
DATA_ROOT = Path(__file__).parent

# Define 7 emotion classes (standardized)
EMOTION_CLASSES = ['afraid', 'angry', 'disgusted', 'happy', 'neutral', 'sad', 'surprised']

# RAF-DB number to emotion mapping (matches folder names)
RAFDB_MAPPING = {
    '1': 'surprised',
    '2': 'fear',
    '3': 'disgusted',
    '4': 'happy',
    '5': 'sad',
    '6': 'angry',
    '7': 'neutral'
}

# Standardize emotion names to match KDEF's 7 classes
EMOTION_STANDARDIZATION = {
    'fear': 'afraid',  # RAF-DB uses 'fear', standardize to 'afraid'
    'afraid': 'afraid',
    'angry': 'angry',
    'disgusted': 'disgusted',
    'happy': 'happy',
    'neutral': 'neutral',
    'sad': 'sad',
    'surprised': 'surprised'
}

# AffectNet to 7-class mapping
AFFECTNET_MAPPING = {
    'anger': 'angry',
    'disgust': 'disgusted',
    'fear': 'afraid',
    'happy': 'happy',
    'neutral': 'neutral',
    'sad': 'sad',
    'surprise': 'surprised',
    'contempt': None  # Ignored
}

# LFW to 7-class mapping (probabilistic)
LFW_MAPPING = {
    'positive': ['happy', 'surprised'],
    'negative': ['angry', 'disgusted', 'sad', 'afraid'],
    'neutral': ['neutral']
}


def get_kdef_pairs() -> List[Tuple[str, str, str]]:
    """
    Extract pairs from KDEF dataset.
    
    Returns:
        List of tuples: (non_occluded_path, occluded_path, emotion_class)
    """
    print("\n" + "="*60)
    print("Processing KDEF Dataset")
    print("="*60)
    
    pairs = []
    kdef_root = DATA_ROOT / "KDEF"
    non_occluded_root = kdef_root / "KDEF_Sorted_Resized"
    occluded_root = kdef_root / "KDEF_Sorted_Occluded"
    
    # Mapping from folder names to standardized emotions
    kdef_emotion_map = {
        'afraid': 'afraid',
        'angry': 'angry',
        'disgusted': 'disgusted',
        'happy': 'happy',
        'neutral': 'neutral',
        'sad': 'sad',
        'surprised': 'surprised'
    }
    
    for emotion_folder in kdef_emotion_map.keys():
        emotion_class = kdef_emotion_map[emotion_folder]
        non_occ_folder = non_occluded_root / emotion_folder
        occ_folder = occluded_root / emotion_folder / "img"
        
        if not non_occ_folder.exists() or not occ_folder.exists():
            print(f"  WARNING: Missing folder for {emotion_folder}")
            continue
        
        # Get all files
        non_occ_files = {f.name: f for f in non_occ_folder.glob("*.jpg")}
        occ_files = {f.name: f for f in occ_folder.glob("*.jpg")}
        
        # Match by filename
        matched = 0
        for filename in non_occ_files.keys():
            if filename in occ_files:
                pairs.append((
                    str(non_occ_files[filename].relative_to(DATA_ROOT)),
                    str(occ_files[filename].relative_to(DATA_ROOT)),
                    emotion_class
                ))
                matched += 1
        
        print(f"  {emotion_class:12s}: {matched:4d} pairs")
    
    print(f"\nTotal KDEF pairs: {len(pairs)}")
    return pairs


def get_rafdb_pairs() -> List[Tuple[str, str, str]]:
    """
    Extract pairs from RAF-DB dataset with quality filtering.
    
    Filtering criteria based on face_visibility_with_features_converted.csv:
    - Face visibility > 60%
    - Landmarks detected
    - Eye condition: (at least one eye > 90% visible) OR (both eyes > 50% visible)
    - If eye condition fails: mouth > 55% visible
    
    Returns:
        List of tuples: (non_occluded_path, occluded_path, emotion_class)
    """
    print("\n" + "="*60)
    print("Processing RAF-DB Dataset (with quality filtering)")
    print("="*60)
    
    pairs = []
    rafdb_root = DATA_ROOT / "RAFDB"
    non_occluded_root = rafdb_root / "RAF-DB"
    occluded_root = rafdb_root / "RAF-DB_Occluded"
    
    # Load the visibility CSV
    visibility_csv = rafdb_root / "face_visibility_with_features_converted.csv"
    
    if not visibility_csv.exists():
        print(f"  WARNING: Visibility CSV not found at {visibility_csv}")
        print(f"  Proceeding without quality filtering...")
        use_filtering = False
        visibility_df = None
    else:
        visibility_df = pd.read_csv(visibility_csv)
        use_filtering = True
        print(f"  Loaded visibility data: {len(visibility_df)} entries")
    
    stats = {
        'total_files': 0,
        'accepted': 0,
        'rejected_face_visibility': 0,
        'rejected_no_landmarks': 0,
        'rejected_eyes_and_mouth': 0,
        'no_visibility_data': 0
    }
    
    for split in ['train', 'test']:
        for emotion_num in RAFDB_MAPPING.keys():
            emotion_name = RAFDB_MAPPING[emotion_num]
            emotion_class = EMOTION_STANDARDIZATION.get(emotion_name, emotion_name)
            
            # Both RAF-DB and RAF-DB_Occluded now use format: "1_surprised", "2_fear", etc.
            emotion_folder_name = f"{emotion_num}_{emotion_name}"
            non_occ_folder = non_occluded_root / split / emotion_folder_name
            occ_folder = occluded_root / split / emotion_folder_name / "hands" / "img"
            
            if not non_occ_folder.exists() or not occ_folder.exists():
                print(f"  WARNING: Missing folder for {split}/{emotion_folder_name}")
                continue
            
            # Get all files
            non_occ_files = {f.name: f for f in non_occ_folder.glob("*.png")}
            
            # Match by filename (handling extension differences: .png to .jpg)
            matched = 0
            for filename in non_occ_files.keys():
                stats['total_files'] += 1
                
                # Convert .png to .jpg for matching
                jpg_name = filename.replace('.png', '.jpg')
                occ_file = occ_folder / jpg_name
                
                if not occ_file.exists():
                    continue
                
                # Apply quality filtering if CSV is available
                if use_filtering:
                    # Find the entry in visibility CSV
                    entry = visibility_df[
                        (visibility_df['emotion'] == int(emotion_num)) &
                        (visibility_df['image_name'] == jpg_name)
                    ]
                    
                    if entry.empty:
                        stats['no_visibility_data'] += 1
                        continue
                    
                    row = entry.iloc[0]
                    
                    # Check if face is detected and landmarks are detected
                    if not row['face_detected'] or not row['landmarks_detected']:
                        stats['rejected_no_landmarks'] += 1
                        continue
                    
                    # Check face visibility > 60%
                    face_vis = row['face_visibility_percentage']
                    if pd.isna(face_vis) or face_vis <= 60:
                        stats['rejected_face_visibility'] += 1
                        continue
                    
                    # Get eye and mouth visibility percentages
                    left_eye = row['left_eye_visible_pct'] if not pd.isna(row['left_eye_visible_pct']) else 0
                    right_eye = row['right_eye_visible_pct'] if not pd.isna(row['right_eye_visible_pct']) else 0
                    mouth = row['mouth_visible_pct'] if not pd.isna(row['mouth_visible_pct']) else 0
                    
                    # Eye condition: (at least one eye > 90%) OR (both eyes > 50%)
                    eye_condition = (left_eye > 90 or right_eye > 90) or (left_eye > 50 and right_eye > 50)
                    
                    # If eye condition fails, check mouth > 55%
                    if not eye_condition and mouth <= 55:
                        stats['rejected_eyes_and_mouth'] += 1
                        continue
                    
                    stats['accepted'] += 1
                
                # Add the pair
                pairs.append((
                    str(non_occ_files[filename].relative_to(DATA_ROOT)),
                    str(occ_file.relative_to(DATA_ROOT)),
                    emotion_class
                ))
                matched += 1
            
            if matched > 0:
                print(f"  {split}/{emotion_class:12s}: {matched:4d} pairs")
    
    # Print filtering statistics
    if use_filtering:
        print(f"\nQuality Filtering Statistics:")
        print(f"  Total files processed: {stats['total_files']}")
        print(f"  Accepted: {stats['accepted']} ({stats['accepted']/stats['total_files']*100:.1f}%)")
        print(f"  Rejected - face visibility <= 60%: {stats['rejected_face_visibility']}")
        print(f"  Rejected - no landmarks: {stats['rejected_no_landmarks']}")
        print(f"  Rejected - eyes and mouth conditions: {stats['rejected_eyes_and_mouth']}")
        print(f"  No visibility data: {stats['no_visibility_data']}")
    
    print(f"\nTotal RAF-DB pairs: {len(pairs)}")
    return pairs


def get_lfw_pairs() -> List[Tuple[str, str, str]]:
    """
    Extract pairs from LFW dataset and map to 7 emotion classes.
    
    Returns:
        List of tuples: (non_occluded_path, occluded_path, emotion_class)
    """
    print("\n" + "="*60)
    print("Processing LFW Dataset")
    print("="*60)
    
    pairs = []
    lfw_root = DATA_ROOT / "LFW"
    non_occluded_root = lfw_root / "M-LFW-FER"
    mask_root = lfw_root / "M-LFW-FER-masks"
    
    lfw_class_counts = defaultdict(int)
    
    for split in ['train', 'eval']:
        for lfw_class in ['positive', 'negative', 'neutral']:
            non_occ_folder = non_occluded_root / split / lfw_class
            mask_folder = mask_root / split / lfw_class
            
            if not non_occ_folder.exists() or not mask_folder.exists():
                print(f"  WARNING: Missing folder for {split}/{lfw_class}")
                continue
            
            # Get all image files
            non_occ_files = list(non_occ_folder.glob("*.jpg"))
            
            for img_file in non_occ_files:
                # Construct mask filename
                mask_filename = img_file.stem + "_mask.png"
                mask_file = mask_folder / mask_filename
                
                if mask_file.exists():
                    # Map LFW class to 7-class emotion
                    possible_emotions = LFW_MAPPING[lfw_class]
                    emotion_class = random.choice(possible_emotions)
                    
                    pairs.append((
                        str(img_file.relative_to(DATA_ROOT)),
                        str(mask_file.relative_to(DATA_ROOT)),
                        emotion_class
                    ))
                    lfw_class_counts[emotion_class] += 1
    
    print("\n  Mapped LFW classes to 7 emotions:")
    for emotion in EMOTION_CLASSES:
        if emotion in lfw_class_counts:
            print(f"    {emotion:12s}: {lfw_class_counts[emotion]:4d} pairs")
    
    print(f"\nTotal LFW pairs: {len(pairs)}")
    return pairs


def get_affectnet_samples(n_samples_per_class: int) -> List[Tuple[str, str, str]]:
    """
    Sample from AffectNet dataset (self-paired for non-occluded training).
    
    Args:
        n_samples_per_class: Number of samples to get per emotion class
        
    Returns:
        List of tuples: (non_occluded_path, occluded_path, emotion_class)
    """
    print("\n" + "="*60)
    print("Processing AffectNet Dataset")
    print("="*60)
    print(f"Target samples per class: {n_samples_per_class}")
    
    pairs = []
    affectnet_root = DATA_ROOT / "AffectNet"
    
    for affectnet_class, mapped_emotion in AFFECTNET_MAPPING.items():
        if mapped_emotion is None:
            continue
        
        class_folder = affectnet_root / affectnet_class
        
        if not class_folder.exists():
            print(f"  WARNING: Missing folder for {affectnet_class}")
            continue
        
        # Get all image files
        image_files = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png"))
        
        # Sample n_samples_per_class (or all if less available)
        n_to_sample = min(n_samples_per_class, len(image_files))
        sampled_files = random.sample(image_files, n_to_sample)
        
        for img_file in sampled_files:
            # Self-paired (same image for both non-occluded and "occluded")
            rel_path = str(img_file.relative_to(DATA_ROOT))
            pairs.append((rel_path, rel_path, mapped_emotion))
        
        print(f"  {mapped_emotion:12s}: {n_to_sample:4d} samples (from {len(image_files)} available)")
    
    print(f"\nTotal AffectNet samples: {len(pairs)}")
    return pairs


def balance_samples_by_emotion(pairs: List[Tuple[str, str, str]], 
                               target_per_class: int = None,
                               use_percentile: float = None) -> List[Tuple[str, str, str]]:
    """
    Balance samples across emotion classes with flexible targeting.
    
    Args:
        pairs: List of (non_occluded_path, occluded_path, emotion_class)
        target_per_class: Target number per class (None = use percentile or minimum)
        use_percentile: Use this percentile of class counts (e.g., 0.6 for 60th percentile)
        
    Returns:
        Balanced list of pairs
    """
    # Group by emotion
    emotion_groups = defaultdict(list)
    for pair in pairs:
        emotion_groups[pair[2]].append(pair)
    
    # Determine target
    if target_per_class is None:
        counts = [len(emotion_groups[e]) for e in EMOTION_CLASSES if e in emotion_groups]
        if use_percentile is not None:
            # Use percentile for more flexible balancing
            target_per_class = int(np.percentile(counts, use_percentile * 100))
        else:
            target_per_class = min(counts) if counts else 0
    
    # Sample from each emotion
    balanced_pairs = []
    for emotion in EMOTION_CLASSES:
        if emotion in emotion_groups:
            available = emotion_groups[emotion]
            n_to_sample = min(target_per_class, len(available))
            balanced_pairs.extend(random.sample(available, n_to_sample))
    
    return balanced_pairs


def split_train_test(pairs: List[Tuple[str, str, str]], 
                     test_ratio: float = 0.2) -> Tuple[List, List]:
    """
    Stratified train/test split.
    
    Args:
        pairs: List of (non_occluded_path, occluded_path, emotion_class)
        test_ratio: Proportion for test set
        
    Returns:
        (train_pairs, test_pairs)
    """
    # Group by emotion
    emotion_groups = defaultdict(list)
    for pair in pairs:
        emotion_groups[pair[2]].append(pair)
    
    train_pairs = []
    test_pairs = []
    
    for emotion, group in emotion_groups.items():
        # Shuffle
        shuffled = group.copy()
        random.shuffle(shuffled)
        
        # Split
        n_test = int(len(shuffled) * test_ratio)
        test_pairs.extend(shuffled[:n_test])
        train_pairs.extend(shuffled[n_test:])
    
    return train_pairs, test_pairs


def print_statistics(pairs: List[Tuple[str, str, str, str, str]], title: str = "Statistics"):
    """Print dataset statistics."""
    print("\n" + "="*60)
    print(title)
    print("="*60)
    
    df = pd.DataFrame(pairs, columns=['non_occluded_path', 'occluded_path', 
                                      'emotion_class', 'dataset_source', 'split'])
    
    print(f"\nTotal samples: {len(df)}")
    print(f"\nBy split:")
    print(df['split'].value_counts().sort_index())
    
    print(f"\nBy dataset:")
    print(df['dataset_source'].value_counts())
    
    print(f"\nBy emotion class:")
    print(df['emotion_class'].value_counts().sort_index())
    
    print(f"\nBy emotion class and split:")
    cross_tab = pd.crosstab(df['emotion_class'], df['split'])
    print(cross_tab)


def main():
    """Main execution function."""
    print("="*60)
    print("Occluded Face Dataset Pairing Script")
    print("="*60)
    
    # Step 1: Extract pairs from each dataset
    kdef_pairs = get_kdef_pairs()
    rafdb_pairs = get_rafdb_pairs()
    lfw_pairs = get_lfw_pairs()
    
    # Combine base datasets
    base_pairs = kdef_pairs + rafdb_pairs + lfw_pairs
    
    # Step 2: Balance base dataset pairs with flexible strategy
    print("\n" + "="*60)
    print("Balancing Base Dataset Samples (Flexible)")
    print("="*60)
    
    # Group by emotion to see distribution
    emotion_counts = defaultdict(int)
    for pair in base_pairs:
        emotion_counts[pair[2]] += 1
    
    print("\nEmotion distribution in base datasets:")
    for emotion in EMOTION_CLASSES:
        print(f"  {emotion:12s}: {emotion_counts.get(emotion, 0):4d} pairs")
    
    # Use 55th percentile for balancing
    # This provides reasonable balance while aiming for 12-13k samples
    counts_list = list(emotion_counts.values())
    target_per_class = int(np.percentile(counts_list, 55))
    
    print(f"\nBalancing strategy:")
    print(f"  Minimum class count: {min(counts_list)}")
    print(f"  Maximum class count: {max(counts_list)}")
    print(f"  55th percentile: {target_per_class}")
    print(f"  Using 55th percentile to target 12-13k samples")
    
    balanced_base_pairs = balance_samples_by_emotion(base_pairs, target_per_class=target_per_class)
    
    # Step 3: Calculate AffectNet samples (35% of base = 0.4667 * base)
    n_base = len(balanced_base_pairs)
    n_affectnet_total = int(0.4667 * n_base)
    n_affectnet_per_class = n_affectnet_total // 7
    
    print(f"\n" + "="*60)
    print(f"Calculating AffectNet Samples")
    print(f"="*60)
    print(f"Base dataset pairs: {n_base}")
    print(f"AffectNet total (35% of base): {n_affectnet_total}")
    print(f"AffectNet per class: {n_affectnet_per_class}")
    
    affectnet_samples = get_affectnet_samples(n_affectnet_per_class)
    
    # Step 4: Combine all pairs
    all_pairs = balanced_base_pairs + affectnet_samples
    
    print(f"\n" + "="*60)
    print(f"Combined Dataset")
    print(f"="*60)
    print(f"Total pairs before split: {len(all_pairs)}")
    print(f"  Base datasets: {len(balanced_base_pairs)} ({len(balanced_base_pairs)/len(all_pairs)*100:.1f}%)")
    print(f"  AffectNet: {len(affectnet_samples)} ({len(affectnet_samples)/len(all_pairs)*100:.1f}%)")
    
    # Step 5: Train/Test split (80:20)
    train_pairs, test_pairs = split_train_test(all_pairs, test_ratio=0.2)
    
    print(f"\n" + "="*60)
    print(f"Train/Test Split")
    print(f"="*60)
    print(f"Train: {len(train_pairs)} ({len(train_pairs)/len(all_pairs)*100:.1f}%)")
    print(f"Test:  {len(test_pairs)} ({len(test_pairs)/len(all_pairs)*100:.1f}%)")
    
    # Step 6: Add dataset source and split labels
    final_data = []
    
    for pair in train_pairs:
        # Determine source dataset
        path = pair[0]
        if 'KDEF' in path:
            source = 'KDEF'
        elif 'RAFDB' in path or 'RAF-DB' in path:
            source = 'RAFDB'
        elif 'LFW' in path:
            source = 'LFW'
        elif 'AffectNet' in path:
            source = 'AffectNet'
        else:
            source = 'Unknown'
        
        final_data.append(pair + (source, 'train'))
    
    for pair in test_pairs:
        # Determine source dataset
        path = pair[0]
        if 'KDEF' in path:
            source = 'KDEF'
        elif 'RAFDB' in path or 'RAF-DB' in path:
            source = 'RAFDB'
        elif 'LFW' in path:
            source = 'LFW'
        elif 'AffectNet' in path:
            source = 'AffectNet'
        else:
            source = 'Unknown'
        
        final_data.append(pair + (source, 'test'))
    
    # Step 7: Save to CSV
    output_file = DATA_ROOT / "dataset_pairs.csv"
    df = pd.DataFrame(final_data, 
                     columns=['non_occluded_path', 'occluded_path', 
                             'emotion_class', 'dataset_source', 'split'])
    
    df.to_csv(output_file, index=False)
    print(f"\nSaved to: {output_file}")
    
    # Print final statistics
    print_statistics(final_data, title="Final Dataset Statistics")
    
    # Also save separate train and test CSVs
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']
    
    train_file = DATA_ROOT / "dataset_pairs_train.csv"
    test_file = DATA_ROOT / "dataset_pairs_test.csv"
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"\nSaved train set to: {train_file}")
    print(f"Saved test set to: {test_file}")
    
    print("\n" + "="*60)
    print("Dataset pairing completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()

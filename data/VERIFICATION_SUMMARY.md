# Dataset Pairing Script Verification Summary

## Folder Structure Verification

### KDEF Dataset
- **Non-occluded path**: `data/KDEF/KDEF_Sorted_Resized/{emotion}/`
  - Emotions: afraid, angry, disgusted, happy, neutral, sad, surprised
  - File format: `*.jpg` (e.g., AF01AFHL.jpg)
  
- **Occluded path**: `data/KDEF/KDEF_Sorted_Occluded/{emotion}/img/`
  - Same emotion folders (lowercase)
  - File format: `*.jpg` with same filenames
  - Also contains `mask/` and `occlusion_mask/` subfolders

### RAF-DB Dataset
- **Non-occluded path**: `data/RAFDB/RAF-DB/{train|test}/{number}_{emotion}/`
  - Folder names: `1_surprised`, `2_fear`, `3_disgusted`, `4_happy`, `5_sad`, `6_angry`, `7_neutral`
  - File format: `*.png` (e.g., train_00006_aligned.png)
  
- **Occluded path**: `data/RAFDB/RAF-DB_Occluded/{train|test}/{number}_{emotion}/hands/img/`
  - Folder names: `1_surprised`, `2_fear`, `3_disgusted`, `4_happy`, `5_sad`, `6_angry`, `7_neutral`
  - File format: `*.jpg` (e.g., train_00006_aligned.jpg)
  - Note: Extension changes from .png to .jpg

- **Quality Filtering**: Uses `data/RAFDB/face_visibility_with_features_converted.csv`
  - **Acceptance Criteria**:
    1. Face visibility > 60%
    2. Face and landmarks detected
    3. Eye condition: (one eye > 90% visible) OR (both eyes > 50% visible)
    4. Fallback: If eyes fail, mouth > 55% visible
  - **Purpose**: Ensures sufficient facial features are visible for emotion recognition
  - **Statistics**: Reports acceptance/rejection counts and reasons

### LFW Dataset
- **Non-occluded path**: `data/LFW/M-LFW-FER/{train|eval}/{class}/`
  - Classes: positive, negative, neutral
  - File format: `*.jpg` (e.g., Aaron_Guiel_0001.jpg)
  
- **Mask path**: `data/LFW/M-LFW-FER-masks/{train|eval}/{class}/`
  - File format: `*.png` with `_mask` suffix (e.g., Aaron_Guiel_0001_mask.png)

### AffectNet Dataset
- **Path**: `data/AffectNet/{emotion}/`
  - Emotions: anger, contempt, disgust, fear, happy, neutral, sad, surprise
  - File format: `*.jpg` and `*.png` mixed
  - Note: contempt is ignored (not in 7-class system)

## Key Script Features

### 1. Correct Path Handling
- KDEF: Uses lowercase emotion folder names directly (not uppercase)
- RAF-DB: Handles folder naming with emotion suffixes for non-occluded
- RAF-DB: Handles extension conversion (.png to .jpg)
- LFW: Properly constructs mask filenames with `_mask` suffix
- AffectNet: Handles both .jpg and .png extensions

### 2. Emotion Class Mapping
```
RAF-DB Mapping:
  1 -> surprised
  2 -> afraid
  3 -> disgusted
  4 -> happy
  5 -> sad
  6 -> angry
  7 -> neutral

AffectNet Mapping:
  anger -> angry
  disgust -> disgusted
  fear -> afraid
  happy -> happy
  neutral -> neutral
  sad -> sad
  surprise -> surprised
  contempt -> None (ignored)

LFW Mapping (probabilistic):
  positive -> [happy, surprised]
  negative -> [angry, disgusted, sad, afraid]
  neutral -> [neutral]
```

### 3. Pairing Strategy
- **KDEF**: Match by exact filename (both .jpg)
- **RAF-DB**: Match by filename, handling .png to .jpg conversion
- **LFW**: Match by adding "_mask.png" suffix
- **AffectNet**: Self-paired (same path for both columns)

### 4. Sampling Strategy
- Pool all images from train/test/eval folders together
- Balance to minimum class count across 7 emotions
- Add AffectNet samples: 35% of base dataset size
- Final ratio: ~68% base datasets, ~32% AffectNet
- Stratified 80:20 train/test split

### 5. Output Format
Three CSV files generated:
1. `dataset_pairs.csv` - Complete dataset
2. `dataset_pairs_train.csv` - Training set only
3. `dataset_pairs_test.csv` - Test set only

Columns:
- `non_occluded_path`: Path to non-occluded image
- `occluded_path`: Path to occluded/masked image
- `emotion_class`: One of 7 emotions
- `dataset_source`: KDEF, RAFDB, LFW, or AffectNet
- `split`: train or test

## Code Quality
- No emojis or special Unicode symbols in output
- Professional console output with clear section headers
- Proper error handling with WARNING messages
- Reproducible results (RANDOM_SEED = 42)
- Type hints for better code clarity
- Comprehensive statistics reporting

## Verification Status
- [x] Syntax check passed
- [x] Folder paths verified against actual structure
- [x] File naming patterns verified
- [x] Extension handling verified
- [x] Emotion mapping verified
- [x] All special symbols removed from code

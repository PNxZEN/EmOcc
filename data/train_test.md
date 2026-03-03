# Dataset Pairing and Train/Test Split Strategy

## Overview
This document outlines the strategy for creating occluded/non-occluded face pairs from multiple facial expression recognition datasets (KDEF, RAF-DB, LFW, and AffectNet) with balanced sampling and proper train/test splitting.

## Dataset Information

### 1. KDEF Dataset
- **Structure**: 
  - Non-occluded: `KDEF/KDEF_Sorted_Resized/` (7 emotion folders)
  - Occluded: `KDEF/KDEF_Sorted_Occluded/{emotion}/img/` (occluded face images)
  - Binary masks: `KDEF/KDEF_Sorted_Occluded/{emotion}/occlusion_mask/` (PNG masks)
- **Emotion Classes** (7): afraid, angry, disgusted, happy, neutral, sad, surprised
- **Pairing Strategy**: Match images by filename between non-occluded and occluded versions
- **Mask Format**: PNG files with white (255) for occluded regions, black (0) for visible regions
- **Mask Filename**: Same as image filename but with .png extension
- **Note**: The `mask` folder exists but is NOT used; only `occlusion_mask` folder is used

### 2. RAF-DB Dataset
- **Structure**:
  - Non-occluded: `RAFDB/RAF-DB/train/` and `RAFDB/RAF-DB/test/` (numbered 1-7)
  - Occluded: `RAFDB/RAF-DB_Occluded/{split}/{emotion}/hands/img/` (occluded face images)
  - Binary masks: `RAFDB/RAF-DB_Occluded/{split}/{emotion}/hands/occlusion_mask/` (PNG masks)
- **Emotion Classes** (7 - numbered):
  - 1 = Surprise
  - 2 = Fear
  - 3 = Disgust
  - 4 = Happiness
  - 5 = Sadness
  - 6 = Anger
  - 7 = Neutral
- **Pairing Strategy**: Match images by filename between RAF-DB and RAF-DB_Occluded
- **Mask Format**: PNG files with white (255) for occluded regions, black (0) for visible regions
- **Mask Filename**: Same as image filename but with .png extension (images are .jpg)
- **Quality Filtering**: Uses `face_visibility_with_features_converted.csv` to filter images
  - **Acceptance Criteria**:
    1. Face visibility > 60%
    2. Landmarks detected (face_detected = True AND landmarks_detected = True)
    3. Eye visibility check:
       - EITHER: At least one eye visible > 90%
       - OR: Both eyes visible > 50%
    4. Fallback condition (if eyes condition fails):
       - Mouth visible > 55%
  - **Rejection Criteria**:
    - Face visibility <= 60%
    - No face detected or no landmarks detected
    - Eyes condition fails AND mouth visibility <= 55%
  - **Rationale**: Ensures occluded faces still have sufficient facial features visible for emotion recognition, particularly focusing on eyes (primary emotion indicators) with mouth as fallback

### 3. LFW Dataset
- **Structure**:
  - Non-occluded: `LFW/M-LFW-FER/M-LFW-FER/train/` and `LFW/M-LFW-FER/M-LFW-FER/eval/`
  - Masks: `LFW/M-LFW-FER-masks/train/` and `LFW/M-LFW-FER-masks/eval/`
- **Original Classes** (3): positive, negative, neutral
- **Class Mapping to 7 Emotions**:
  - **Positive** → Randomly distributed to: happy, surprised (positive emotions)
  - **Negative** → Randomly distributed to: angry, disgusted, sad, afraid/fear (negative emotions)
  - **Neutral** → neutral
- **Pairing Strategy**: Match images with corresponding mask files (image.jpg ↔ image_mask.png)
- **Mask Format**: PNG overlay images where white/bright regions indicate occlusion
- **Note**: LFW masks are overlay images, not pure binary masks, but processed the same way

### 4. AffectNet Dataset
- **Structure**: `AffectNet/` (8 emotion folders)
- **Emotion Classes** (8): anger, contempt, disgust, fear, happy, neutral, sad, surprise
- **Mapping to 7 Classes**:
  - anger → angry
  - disgust → disgusted
  - fear → afraid
  - happy → happy
  - neutral → neutral
  - sad → sad
  - surprise → surprised
  - contempt → ignored (not in 7-class system)
- **Pairing Strategy**: Each image paired with itself (non-occluded = occluded path)
- **Mask Format**: No masks (self-paired clean images for consistency loss and preventing catastrophic forgetting)

## Binary Mask Specifications

### Mask Conventions Across Datasets
All binary masks follow a consistent convention:
- **White pixels (value 255)**: Occluded regions (to be downweighted by attention mechanism)
- **Black pixels (value 0)**: Visible regions (to be preserved)

### Mask-Image Correspondence

**KDEF:**
- Image path: `KDEF/KDEF_Sorted_Occluded/{emotion}/img/{filename}.jpg`
- Mask path: `KDEF/KDEF_Sorted_Occluded/{emotion}/occlusion_mask/{filename}.png`
- Example: `afraid/img/AF01AFHL.jpg` → `afraid/occlusion_mask/AF01AFHL.png`

**RAF-DB:**
- Image path: `RAFDB/RAF-DB_Occluded/{split}/{emotion_folder}/hands/img/{filename}.jpg`
- Mask path: `RAFDB/RAF-DB_Occluded/{split}/{emotion_folder}/hands/occlusion_mask/{filename}.png`
- Example: `train/1_surprised/hands/img/train_00001_aligned.jpg` → `train/1_surprised/hands/occlusion_mask/train_00001_aligned.png`

**LFW:**
- Image path: `LFW/M-LFW-FER-masks/{split}/{class}/{name}_mask.png`
- The LFW dataset stores masked images directly; masks are extracted from the masked images themselves
- Example: `train/positive/Aaron_Eckhart_0001_mask.png` (mask extracted from pixel values)

**AffectNet:**
- No masks needed (self-paired clean images)
- Binary mask is all zeros during training

### Dataset Loader Mask Processing
The `PairedOcclusionDataset` class handles mask loading automatically:
1. Derives mask path from occluded image path based on dataset source
2. Loads mask as grayscale PNG
3. Converts to binary tensor: threshold at 0.5 (white > 0.5 → 1, black ≤ 0.5 → 0)
4. Resizes to 224x224 to match image dimensions
5. Downsamples to 7x7 internally for attention mechanism alignment

### Training Usage
Binary masks are used exclusively during training:
- **Training Phase**: Provided to student model and loss functions
- **Inference Phase**: Not provided; attention mechanism learns to detect occlusion from features alone
- **Purpose**: Guide attention mechanism to learn which spatial regions are occluded

## Processing Steps

### Step 1: Pool All Images
- Ignore existing train/test/eval folder distinctions
- Combine all images from each dataset into a single pool
- Maintain emotion class labels

### Step 2: Create Pairs
For each dataset:
- **KDEF**: Match non-occluded with occluded by filename
- **RAF-DB**: Match non-occluded with occluded by filename, apply quality filtering (see RAF-DB Quality Filtering section below)
- **LFW**: Match images with mask files, map to 7 emotion classes
- Store as tuples: (non_occluded_path, occluded_path, emotion_class, dataset_source)

### Step 2.1: RAF-DB Quality Filtering
Applies strict quality criteria to ensure only high-quality occluded samples are included:

**Primary Criteria (ALL must be satisfied):**
1. Face visibility > 60% of total face area
2. Face detected successfully
3. Facial landmarks detected successfully

**Feature Visibility Criteria (at least ONE must be satisfied):**
1. **Eye Condition (Preferred)**:
   - At least one eye has > 90% visibility (handles cases where one eye is occluded)
   - OR both eyes have > 50% visibility each
2. **Mouth Fallback Condition**:
   - If eye condition fails, mouth must have > 55% visibility
   - Handles cases like sunglasses where eyes are occluded but mouth is visible

**Rationale:**
- Eyes are primary indicators for emotion recognition (especially for fear, surprise, happiness)
- Mouth provides critical backup information (especially for happiness, disgust, sadness)
- Face visibility threshold ensures sufficient facial context
- Prevents including heavily occluded samples that would be too ambiguous for training
- Balances dataset quality with quantity

**Statistics Reported:**
- Total files processed
- Accepted samples (with percentage)
- Rejected due to face visibility <= 60%
- Rejected due to missing face/landmarks detection
- Rejected due to insufficient eye and mouth visibility
- Files without visibility data

### Step 3: Balanced Sampling (7 Emotion Classes)
- Calculate the minimum number of pairs available across all 7 emotion classes
- Sample approximately equal number of pairs from each emotion class
- Target distribution: roughly 1/7th of total pairs per emotion
- Handle LFW mapping probabilistically to maintain balance

### Step 4: Proportion Calculation (75:35 ratio)
- Let `N_base` = total pairs from KDEF + RAF-DB + LFW (this represents 75%)
- Calculate `N_affectnet = (35/75) * N_base = 0.4667 * N_base`
- Sample `N_affectnet` images from AffectNet, balanced across 7 emotions
- Each AffectNet image paired with itself

### Step 5: Combine All Pairs
- Merge pairs from KDEF, RAF-DB, LFW, and AffectNet
- Final ratio: ~68% from main datasets, ~32% from AffectNet

### Step 6: Train/Test Split (80:20)
- Stratified split by emotion class to maintain class balance
- 80% → training set
- 20% → test set
- Use random shuffling with fixed seed for reproducibility

### Step 7: Save to CSV
Output format:
```
non_occluded_path,occluded_path,emotion_class,dataset_source,split
data/KDEF/.../AF01AFHL.jpg,data/KDEF/.../AF01AFHL.jpg,afraid,KDEF,train
...
```

## Rationale

### LFW Class Mapping Justification
**Theoretical Basis**: Valence-based emotion categorization
- **Positive valence emotions** (approach-oriented): happy, surprised
  - Happiness: clearly positive
  - Surprise: generally positive in social contexts (unexpected pleasant events)
  
- **Negative valence emotions** (avoidance-oriented): angry, afraid, disgusted, sad
  - Anger: negative, threat-related
  - Afraid/Fear: negative, threat-related
  - Disgust: negative, avoidance-related
  - Sadness: negative, withdrawal-related
  
- **Neutral valence**: neutral
  - No strong emotional content

This mapping aligns with Russell's circumplex model of affect and basic emotion theory.

### AffectNet Self-Pairing Justification
**Theoretical Basis**: Preventing catastrophic forgetting
- The model needs to maintain ability to recognize non-occluded faces
- Including clean face pairs (same image) ensures the model doesn't forget clean face features
- 35% proportion prevents overfitting to occlusion detection while maintaining baseline performance
- Acts as a regularization technique during training

### Balanced Sampling Justification
- Prevents class imbalance issues during training
- Ensures model performs equally well across all emotions
- Critical for fair evaluation on minority emotion classes

## Implementation Notes
- Use fixed random seed (e.g., 42) for reproducibility
- Verify all file paths exist before creating pairs
- Log statistics: total pairs per dataset, per emotion, per split
- Handle missing files gracefully with warnings

# FED-RO Evaluation System

Comprehensive evaluation framework for comparing Teacher (FECNet) and Student (OccFECNet) models on the FED-RO (Facial Expression Dataset - Romanian Occluded) dataset.

## Overview

The evaluation system provides:
1. **Quantitative Metrics**: Cosine similarity and L2 distance between embeddings
2. **Visual Analysis**: Sample-by-sample comparison with similarity scores
3. **Distribution Analysis**: Overall and per-emotion performance
4. **Embedding Space Comparison**: PCA visualization of learned representations

## Quick Start

### Basic Evaluation
```bash
cd eval
python evaluate_fedro.py \
    --teacher_path ../pretrained/FECNet.pt \
    --student_path ../checkpoints/curriculum/student_best.pth \
    --data_root ../data/FED-RO/FED-RO_crop
```

### Custom Configuration
```bash
python evaluate_fedro.py \
    --teacher_path ../pretrained/FECNet.pt \
    --student_path ../checkpoints/curriculum/student_epoch_80.pth \
    --data_root ../data/FED-RO/FED-RO_crop \
    --batch_size 64 \
    --n_samples 32 \
    --output_dir eval_epoch_80
```

## Output Files

### 1. evaluation_metrics.txt
Detailed quantitative metrics:
- **Overall Statistics**:
  - Mean/Std/Min/Max cosine similarity
  - Mean/Std/Min/Max L2 distance
- **Per-Emotion Breakdown**:
  - Count, mean, and standard deviation for each emotion
- **Interpretation Guide**:
  - Threshold-based quality assessment

Example:
```
Overall Metrics:
Cosine Similarity:
  Mean:  0.932451
  Std:   0.041234
  Min:   0.781234
  Max:   0.987654

Per-Emotion Metrics:
Emotion      Count    Cos Sim Mean    Cos Sim Std     L2 Dist Mean    L2 Dist Std
Anger        150      0.928123        0.045234        0.512345        0.123456
...
```

### 2. similarity_distribution.png
Two-panel visualization:
- **Left**: Histogram of cosine similarities across all samples
- **Right**: Bar chart of per-emotion mean similarities with error bars

### 3. sample_comparison_best.png
Grid of 8 images with **highest** teacher-student similarity:
- Shows emotion label
- Cosine similarity score
- L2 distance
- Color-coded titles (green=excellent, orange=good, red=poor)

### 4. sample_comparison_worst.png
Grid of 8 images with **lowest** teacher-student similarity:
- Helps identify failure cases
- Useful for debugging and model improvement

### 5. embedding_space_pca.png
Side-by-side PCA projections:
- **Left**: Teacher embedding space
- **Right**: Student embedding space
- Color-coded by emotion
- Shows how well student preserves teacher's embedding structure

### 6. fedro_embeddings.npz
NumPy archive containing:
- `teacher_embeddings`: [N, 16] teacher embeddings
- `student_embeddings`: [N, 16] student embeddings
- `emotions`: [N] emotion labels
- `emotion_indices`: [N] emotion indices
- `paths`: [N] image paths

## Metrics Explained

### Cosine Similarity
Measures angular similarity between embeddings (range: -1 to 1, higher is better)

**Interpretation**:
- **> 0.95**: Excellent - Student very closely matches teacher
- **0.90 - 0.95**: Good - Minor differences, acceptable performance
- **0.85 - 0.90**: Fair - Noticeable differences but still aligned
- **< 0.85**: Poor - Significant divergence, needs investigation

**What it means**:
- High similarity: Student learned to produce similar representations to teacher
- Low similarity: Student may have learned different features or failed to distill properly

### L2 Distance
Euclidean distance between embeddings (range: 0 to ∞, lower is better)

**Interpretation**:
- **< 0.5**: Very close - Nearly identical embeddings
- **0.5 - 1.0**: Moderate - Acceptable difference
- **> 1.0**: Large - Significant difference in embedding space

**What it means**:
- Small distance: Embeddings are numerically close
- Large distance: Embeddings differ substantially in magnitude/direction

## Understanding Results

### Good Student Model Indicators:
1. **High mean cosine similarity** (>0.90)
2. **Low standard deviation** (<0.05) - consistent across samples
3. **Balanced per-emotion performance** - all emotions >0.85
4. **Similar PCA projections** - student clusters match teacher
5. **Most samples in "best" category** - few failure cases

### Red Flags:
1. **Low mean similarity** (<0.85) - poor distillation
2. **High variance** (>0.10) - inconsistent performance
3. **Emotion-specific drops** - certain emotions much worse
4. **Different PCA structure** - student learned different features
5. **Many "worst" cases** - frequent failures

### Per-Emotion Analysis:
- **Anger/Disgust/Fear**: Often harder due to occlusion hiding key features
- **Happy**: Usually easier, smile is distinctive even with occlusion
- **Neural (Neutral)**: Baseline - should have good similarity
- **Sad**: Moderate difficulty, subtle expression
- **Surprise**: Usually good, wide eyes/open mouth distinctive

## Advanced Usage

### Compare Multiple Checkpoints
```bash
# Evaluate different epochs
for epoch in 40 60 80 100; do
    python evaluate_fedro.py \
        --student_path ../checkpoints/curriculum/student_epoch_${epoch}.pth \
        --output_dir eval_epoch_${epoch}
done

# Compare results
python compare_checkpoints.py --dirs eval_epoch_*
```

### Load and Analyze Saved Embeddings
```python
import numpy as np
import matplotlib.pyplot as plt

# Load embeddings
data = np.load('fedro_embeddings.npz', allow_pickle=True)
teacher_emb = data['teacher_embeddings']
student_emb = data['student_embeddings']
emotions = data['emotions']

# Compute per-sample similarity
cos_sim = (teacher_emb * student_emb).sum(axis=1) / (
    np.linalg.norm(teacher_emb, axis=1) * np.linalg.norm(student_emb, axis=1)
)

# Find problematic samples
low_sim_indices = np.where(cos_sim < 0.80)[0]
print(f"Samples with similarity < 0.80: {len(low_sim_indices)}")

# Analyze by emotion
for emotion in np.unique(emotions):
    mask = emotions == emotion
    print(f"{emotion}: {cos_sim[mask].mean():.4f} ± {cos_sim[mask].std():.4f}")
```

### Extract Specific Emotion Performance
```python
# From evaluation_metrics.txt
import re

with open('evaluation_metrics.txt', 'r') as f:
    content = f.read()
    
# Extract happy emotion metrics
happy_pattern = r"Happy\s+(\d+)\s+([\d.]+)\s+([\d.]+)"
match = re.search(happy_pattern, content)
if match:
    count, mean_sim, std_sim = match.groups()
    print(f"Happy: {count} samples, {mean_sim} ± {std_sim}")
```

## Troubleshooting

### Issue: Out of memory
**Solution**: Reduce batch size
```bash
python evaluate_fedro.py --batch_size 16
```

### Issue: Wrong emotion labels
**Note**: FED-RO uses 'neural' not 'neutral'
- The script handles this automatically
- Check that your data is in `FED-RO_crop/` not `FED-RO_original/`

### Issue: Missing images
**Solution**: Verify data structure
```bash
# Should show: anger, disgust, fear, happy, neural, sad, surprise
ls data/FED-RO/FED-RO_crop/
```

### Issue: Different results than expected
**Checks**:
1. Verify student checkpoint is correct epoch
2. Confirm teacher model loaded properly
3. Check if using best vs final vs epoch checkpoint
4. Ensure data preprocessing matches training

## Benchmark Standards

Based on typical performance:

### Excellent Student (Target):
- Mean cosine similarity: **> 0.92**
- All emotions: **> 0.88**
- Std deviation: **< 0.05**
- L2 distance: **< 0.6**

### Good Student (Acceptable):
- Mean cosine similarity: **0.88 - 0.92**
- All emotions: **> 0.85**
- Std deviation: **< 0.08**
- L2 distance: **< 0.8**

### Poor Student (Needs Improvement):
- Mean cosine similarity: **< 0.88**
- Any emotion: **< 0.85**
- Std deviation: **> 0.10**
- L2 distance: **> 1.0**

## Expected Runtime

On NVIDIA RTX 3090:
- Embedding extraction: ~30 seconds (FED-RO has ~1000 images)
- Metric computation: ~5 seconds
- Visualization: ~10 seconds
- **Total**: ~1 minute

## Next Steps After Evaluation

### If results are good (>0.90):
✓ Model is well-distilled
✓ Ready for deployment
✓ Can proceed to other datasets (AffectNet, KDEF)

### If results are moderate (0.85-0.90):
- Check attention correlation during training
- Verify beta value progression
- Consider training longer (more Phase 4 epochs)
- Review worst-case samples for patterns

### If results are poor (<0.85):
- Review training logs for issues
- Check if NaN occurred during training
- Verify dataset quality and masks
- Consider adjusting hyperparameters (lambda values)
- May need to retrain from checkpoint before divergence

## References

- FED-RO Dataset: Romanian occluded facial expressions
- Evaluation metrics based on embedding similarity analysis
- PCA visualization for dimensionality reduction

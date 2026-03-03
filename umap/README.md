# UMAP Embedding Space Visualization

This directory contains tools for visualizing and comparing the embedding spaces learned by the teacher (FECNet) and student (OccFECNet) models using UMAP dimensionality reduction on AffectNet dataset.

## Requirements

Install UMAP if not already installed:
```bash
pip install umap-learn
```

## Quick Start

### 1. Basic Visualization (5000 samples)
```bash
python visualize_embedding_space.py \
    --teacher_path ../pretrained/FECNet.pt \
    --student_path ../checkpoints/curriculum/student_best.pth \
    --csv_path ../data/AffectNet/labels.csv \
    --data_root ../data/AffectNet
```

### 2. Custom Sample Size
```bash
# Use all available samples
python visualize_embedding_space.py \
    --teacher_path ../pretrained/FECNet.pt \
    --student_path ../checkpoints/curriculum/student_best.pth \
    --csv_path ../data/AffectNet/labels.csv \
    --data_root ../data/AffectNet \
    --max_samples 10000
```

### 3. Adjust UMAP Parameters
```bash
# More local structure (smaller n_neighbors)
python visualize_embedding_space.py \
    --teacher_path ../pretrained/FECNet.pt \
    --student_path ../checkpoints/curriculum/student_best.pth \
    --csv_path ../data/AffectNet/labels.csv \
    --data_root ../data/AffectNet \
    --n_neighbors 30 \
    --min_dist 0.05 \
    --metric euclidean
```

## Output Files

The script generates the following files in the `umap/` directory:

1. **umap_teacher_vs_student.png**
   - Side-by-side comparison of teacher and student embedding spaces
   - Shows both models' UMAP projections with emotion-based coloring

2. **umap_teacher_detailed.png**
   - High-resolution plot of teacher model embeddings
   - Includes sample counts per emotion in legend

3. **umap_student_detailed.png**
   - High-resolution plot of student model embeddings
   - Includes sample counts per emotion in legend

4. **embedding_statistics.txt**
   - Overall cosine similarity between teacher and student embeddings
   - Per-emotion statistics (mean and std of cosine similarity)

5. **umap_coordinates.npz**
   - NumPy archive containing:
     - `teacher_umap`: Teacher UMAP coordinates [N, 2]
     - `student_umap`: Student UMAP coordinates [N, 2]
     - `teacher_embeddings`: Teacher embeddings [N, 16]
     - `student_embeddings`: Student embeddings [N, 16]
     - `emotions`: Emotion labels [N]
     - `emotion_indices`: Emotion indices [N]

## Understanding the Visualizations

### Emotion Color Coding
- **Anger**: Red
- **Contempt**: Purple
- **Disgust**: Green
- **Fear**: Orange
- **Happy**: Yellow
- **Neutral**: Gray
- **Sad**: Blue
- **Surprise**: Teal

### What to Look For

**Good Student Model:**
- Similar cluster structure to teacher
- Emotions well-separated in both spaces
- High cosine similarity (>0.85) between corresponding teacher-student embeddings

**Potential Issues:**
- Student clusters more scattered than teacher: May need more training
- Different emotion groupings: Model learning different features
- Low cosine similarity (<0.70): Student not properly distilled

## UMAP Parameters

### n_neighbors (default: 15)
- **Small (5-10)**: Emphasizes local structure, creates tighter clusters
- **Medium (15-30)**: Balanced view
- **Large (50+)**: Emphasizes global structure, preserves relationships between distant points

### min_dist (default: 0.1)
- **Small (0.0-0.1)**: Points packed tightly together
- **Medium (0.1-0.5)**: Balanced spacing
- **Large (0.5+)**: Points more evenly distributed

### metric (default: cosine)
- **cosine**: Good for normalized embeddings (like L2-normalized FECNet outputs)
- **euclidean**: Standard Euclidean distance
- **manhattan**: L1 distance

## Advanced Usage

### Load and Analyze Saved Coordinates

```python
import numpy as np
import matplotlib.pyplot as plt

# Load coordinates
data = np.load('umap_coordinates.npz', allow_pickle=True)
teacher_umap = data['teacher_umap']
student_umap = data['student_umap']
emotions = data['emotions']

# Compute distance between teacher and student points
distances = np.linalg.norm(teacher_umap - student_umap, axis=1)
print(f"Mean distance: {distances.mean():.4f}")

# Plot distance distribution
plt.hist(distances, bins=50)
plt.xlabel('UMAP Distance (Teacher vs Student)')
plt.ylabel('Count')
plt.title('Distribution of Point Distances')
plt.show()
```

### Compare Multiple Student Checkpoints

```bash
# Create separate visualizations for different checkpoints
for epoch in 40 60 80 100; do
    python visualize_embedding_space.py \
        --student_path ../checkpoints/curriculum/student_epoch_${epoch}.pth \
        --output_dir umap_epoch_${epoch}
done
```

## Interpreting Results

### High-Quality Student Model Indicators:
1. **Cluster Preservation**: Emotions cluster similarly in both teacher and student spaces
2. **High Cosine Similarity**: Overall mean >0.85
3. **Per-Emotion Consistency**: All emotions show >0.80 similarity
4. **Separation**: Clear boundaries between emotion clusters

### Common Patterns:
- **Happy vs Surprise**: Often close together (positive emotions)
- **Anger vs Disgust**: May overlap (negative high-arousal emotions)
- **Sad vs Neutral**: Can be nearby (low-arousal emotions)
- **Fear**: Sometimes scattered (diverse expressions)

## Troubleshooting

### Issue: Out of memory
**Solution**: Reduce batch_size or max_samples
```bash
python visualize_embedding_space.py --batch_size 32 --max_samples 2000
```

### Issue: UMAP too slow
**Solution**: Use fewer samples or reduce n_neighbors
```bash
python visualize_embedding_space.py --max_samples 3000 --n_neighbors 10
```

### Issue: Clusters too tight/loose
**Solution**: Adjust min_dist parameter
```bash
# Tighter clusters
python visualize_embedding_space.py --min_dist 0.01

# Looser clusters
python visualize_embedding_space.py --min_dist 0.3
```

## Expected Runtime

On NVIDIA RTX 3090 with 5000 samples:
- Embedding extraction: ~2-3 minutes
- UMAP computation: ~1-2 minutes per model
- **Total**: ~5-7 minutes

## References

- UMAP: McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv:1802.03426
- AffectNet: Mollahosseini, A., Hasani, B., & Mahoor, M. H. (2017). AffectNet: A Database for Facial Expression, Valence, and Arousal Computing in the Wild. IEEE Transactions on Affective Computing.

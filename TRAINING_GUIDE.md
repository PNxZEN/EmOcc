# Training Guide for OccFECNet Student Model

## Quick Start

### 1. First Time Training
```bash
python train_curriculum.py \
    --teacher_path pretrained/FECNet.pt \
    --train_csv data/dataset_pairs_train.csv \
    --test_csv data/dataset_pairs_test.csv \
    --experiment_name my_first_run \
    --batch_size 90 \
    --epochs 100
```

### 2. Resume from Checkpoint
```bash
# Resume from specific epoch
python train_curriculum.py \
    --teacher_path pretrained/FECNet.pt \
    --train_csv data/dataset_pairs_train.csv \
    --test_csv data/dataset_pairs_test.csv \
    --resume_from checkpoints/curriculum/student_epoch_25.pth \
    --experiment_name my_first_run_resumed

# Resume from best model
python train_curriculum.py \
    --teacher_path pretrained/FECNet.pt \
    --train_csv data/dataset_pairs_train.csv \
    --test_csv data/dataset_pairs_test.csv \
    --resume_from checkpoints/curriculum/student_best.pth \
    --experiment_name continue_best
```

### 3. Visualize Training Results
```bash
# After training completes
python visualize_training_curves.py \
    --log_dir logs/curriculum \
    --experiment my_first_run

# Compare multiple experiments
python visualize_training_curves.py \
    --log_dir logs/curriculum \
    --compare my_first_run my_second_run my_third_run

# List available experiments
python visualize_training_curves.py \
    --log_dir logs/curriculum \
    --list
```

---

## Training Phases Overview

The training follows a 4-phase curriculum (100 epochs total):

### Phase 1: Baseline (Epochs 1-10)
- **Goal**: Student learns to match teacher on clean faces
- **Input**: Clean faces only
- **Attention**: Frozen (beta=0, transparent)
- **Expected**: Distillation loss approaches 0, cosine similarity approaches 0.99

### Phase 2: Introduce Occlusion (Epochs 11-20)
- **Goal**: Expose student to occlusion while attention remains inactive
- **Input**: 50% clean, 50% occluded
- **Attention**: Still frozen
- **Expected**: Loss increases but converges, beta still ~0

### Phase 3: Activate Attention (Epochs 21-40)
- **Goal**: Teach attention to detect and downweight occlusions
- **Input**: 50% clean, 50% occluded
- **Attention**: UNFROZEN (beta learnable, conv layers trainable)
- **Lambda3**: Progressive 0.0 to 0.05 to 0.1
- **Expected**: beta increases (0.3-0.8), attention-mask correlation 0.6-0.9

### Phase 4: Full Training (Epochs 41-100)
- **Goal**: Finalize with all parameters active
- **Input**: 50% clean, 50% occluded
- **Attention**: Fully active
- **Lambda3**: Fixed at 0.1
- **Expected**: All losses stabilize, high validation performance

---

## Monitoring During Training

### Key Metrics to Watch

#### Phase 1-2:
- **Distillation Loss**: Should decrease toward 0
- **Consistency Loss**: Should be low (Phase 2 only)
- **Beta**: Should remain ~0

#### Phase 3-4:
- **Distillation Loss**: Should decrease as attention helps
- **Beta**: Should increase from 0 toward 0.3-0.8
- **Attention-Mask Correlation**: TARGET 0.6-0.9
  - Too low (<0.6)? Attention not learning, may need higher lambda3
  - Too high (>0.95)? Attention copying mask, may need lower lambda3
- **Attention Entropy**: Should be moderate (not collapsed)

### Checkpoints Saved
- Every 5 epochs: `student_epoch_5.pth`, `student_epoch_10.pth`, etc.
- Phase boundaries: Epoch 10, 20, 40, 100
- Best model: `student_best.pth` (highest cosine similarity)
- Final model: `student_final.pth`

---

## Common Issues and Solutions

### Issue: Training interrupted
**Solution**: Resume from last checkpoint
```bash
python train_curriculum.py \
    --resume_from checkpoints/curriculum/student_epoch_35.pth \
    --teacher_path pretrained/FECNet.pt \
    --train_csv data/dataset_pairs_train.csv \
    --test_csv data/dataset_pairs_test.csv
```

### Issue: Attention not learning (correlation < 0.6 in Phase 3)
**Possible causes**:
1. lambda3 too low, increase lambda3 manually in code
2. Learning rate too low, check optimizer settings
3. Binary masks incorrect, verify mask loading

**Debug**:
```python
# In train_epoch(), add after loss computation:
if config['phase'] >= 3:
    print(f"Attn correlation: {metrics['attn_correlation']:.4f}")
    print(f"Lambda3: {config['lambda3']:.4f}")
```

### Issue: Attention copying masks exactly (correlation > 0.95)
**Solution**: Decrease lambda3 (too much regularization)

### Issue: Beta not increasing in Phase 3
**Possible causes**:
1. Beta learning rate too low (should be 1e-5)
2. Attention still frozen, check `unfreeze_attention()` was called
3. Gradients clipped too aggressively

**Debug**:
```python
# Check beta gradient
print(f"Beta: {student.attention.beta.item():.6f}")
print(f"Beta grad: {student.attention.beta.grad}")
```

### Issue: Out of memory
**Solutions**:
- Reduce batch size: `--batch_size 64` or `--batch_size 45`
- Use gradient accumulation (modify training loop)
- Use mixed precision training (add `torch.cuda.amp`)

---

## Architecture Notes

### Differences from OccFECNet Paper

Our implementation uses **InceptionResnetV1** instead of **FaceNet NN2**:

| Component | Paper (FaceNet NN2) | Our Implementation |
|-----------|-------------------|-------------------|
| Feature maps | 7x7x1024 | 5x5x1792 |
| Attention input | 1025 channels | 1793 channels |
| Attention output | 7x7 map | 5x5 map |

**Why?** InceptionResnetV1 has better pretrained weights (VGGFace2) and more modern architecture. All other aspects follow the paper exactly.

---

## File Structure

```
checkpoints/curriculum/
├── student_epoch_5.pth
├── student_epoch_10.pth (Phase 1 end)
├── student_epoch_15.pth
├── student_epoch_20.pth (Phase 2 end)
├── ...
├── student_epoch_100.pth (Phase 4 end)
├── student_best.pth      # Best validation score
└── student_final.pth     # Final model

logs/curriculum/
├── <experiment_name>_train.csv
└── <experiment_name>_eval.csv

figures/training/
├── <experiment_name>_curves.png
├── <experiment_name>_attention.png
└── comparison.png
```

---

## Advanced: Custom Training Configuration

### Modify Phase Boundaries
Edit `get_phase_config()` in `train_curriculum.py`:
```python
if phase == 1:
    return {
        'epochs': (1, 15),  # Extend Phase 1 to epoch 15
        ...
    }
```

### Adjust Lambda Values
```python
# Phase 3
'lambda3': lambda3 * 1.5,  # Increase attention regularization
```

### Change Learning Rates
```python
# Phase 4
'lr_beta': 1e-4,  # Increase beta learning rate
```

---

## Expected Training Time

On NVIDIA RTX 3090:
- Batch size 90: ~20 hours for 100 epochs
- Batch size 45: ~13 hours for 100 epochs

On NVIDIA V100:
- Batch size 90: ~25 hours for 100 epochs

**Factors**:
- Dataset size (14,618 samples)
- Image augmentation
- Attention computation in Phase 3-4

---

## Next Steps After Training

1. **Evaluate on test set**:
   ```bash
   python evaluate_student.py \
       --model checkpoints/curriculum/student_best.pth \
       --test_csv data/dataset_pairs_test.csv
   ```

2. **Visualize attention maps**:
   ```bash
   python visualize_attention.py \
       --model checkpoints/curriculum/student_best.pth \
       --image path/to/occluded_face.jpg
   ```

3. **Deploy for inference**:
   - Use `student_best.pth` or `student_final.pth`
   - No binary mask needed at inference
   - Attention operates purely from features

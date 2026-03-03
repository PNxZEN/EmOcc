# Teacher vs Student Performance Comparison on Occluded Faces

## Overview

This evaluation demonstrates that the **student model outperforms the teacher on occluded faces** while maintaining comparable performance on clean faces - validating the effectiveness of our curriculum learning approach with spatial attention.

## Key Hypothesis

> The student model, trained with progressive occlusion exposure and spatial attention mechanisms, should show **superior performance on occluded datasets** (FED-RO) while maintaining **comparable accuracy on clean datasets** (KDEF/AffectNet).

## Quick Start

```bash
cd eval
python compare_occluded_performance.py \
    --teacher_path ../pretrained/FECNet.pt \
    --student_path ../checkpoints/curriculum/student_best.pth \
    --fedro_root ../data/FED-RO/FED-RO_crop \
    --kdef_root ../data/KDEF/KDEF_Sorted_Resized
```

**Expected runtime**: ~2-3 minutes on RTX 3090

## What This Shows

### 1. **Accuracy Comparison** (`accuracy_comparison.png`)
- Bar chart comparing teacher vs student on:
  - **FED-RO (occluded)**: Student should show **+X% improvement**
  - **KDEF (clean)**: Student should maintain **comparable accuracy** (±2%)

**What to look for**:
- ✅ Student bar higher on FED-RO → Confirms occlusion robustness
- ✅ Similar bars on KDEF → No degradation on clean faces
- ❌ Teacher higher on FED-RO → Training issue, needs investigation

### 2. **Confusion Matrices** (`confusion_matrices.png`)
- Side-by-side heatmaps on FED-RO dataset
- Shows which emotions are confused

**What to look for**:
- ✅ Brighter diagonal on student → Better per-class accuracy
- ✅ Reduced off-diagonal on student → Fewer misclassifications
- Common confusions: fear/surprise, anger/disgust (expected)

### 3. **Per-Emotion Accuracy** (`per_emotion_accuracy.png`)
- Detailed breakdown for all 7 emotions
- Identifies which emotions benefit most from attention

**What to look for**:
- ✅ Student bars higher across most emotions
- Emotions with large improvement → Most affected by occlusion in teacher
- Typical pattern: Happiness, surprise show largest gains (distinctive features get occluded)

### 4. **Spatial Attention Visualization** (`attention_visualization.png`)
- Shows 8 sample occluded faces with attention overlays
- **This is the key visual proof**

**What to look for**:
- ✅ Red/yellow attention on **visible facial regions**
- ✅ Blue/dark attention on **occluded areas** (hands, objects)
- ✅ Attention focuses on **eyes** when mouth occluded
- ✅ Attention focuses on **mouth** when eyes occluded

### 5. **Detailed Text Report** (`performance_comparison_report.txt`)
- Quantitative metrics
- Per-emotion breakdown
- Key findings summary
- Hypothesis validation

## Expected Results

### Benchmark Standards

| Metric | Expected Value | Interpretation |
|--------|---------------|----------------|
| **FED-RO Improvement** | +5% to +15% | Student outperforms teacher on occluded faces |
| **KDEF Difference** | -2% to +2% | Student maintains clean face performance |
| **Per-Emotion Gains** | Positive for 5+ emotions | Broad improvement across emotions |
| **Attention Quality** | Visual focus on visible regions | Learned to identify occlusion |

### Example Output

```
ACCURACY COMPARISON
============================================================
FED-RO (Occluded):
  Teacher: 68.5%
  Student: 78.3%
  Improvement: +14.3%  ← STUDENT WINS ON OCCLUDED

KDEF (Clean):
  Teacher: 85.2%
  Student: 84.8%
  Difference: -0.5%   ← COMPARABLE ON CLEAN
============================================================
```

## Interpreting Results

### ✅ Success Indicators
1. **FED-RO improvement > +5%**: Student is robustly better on occlusion
2. **KDEF difference within ±3%**: No catastrophic forgetting
3. **Attention highlights visible regions**: Spatial attention works correctly
4. **Broad per-emotion gains**: Not just learning specific emotions

### ⚠️ Warning Signs
1. **FED-RO improvement < +2%**: Student not learning occlusion handling
2. **KDEF drops > -5%**: Overfitting to occluded data, forgetting clean
3. **Attention unfocused**: Random attention patterns indicate poor learning
4. **Gains only in 1-2 emotions**: Overly specialized, not generalizing

### ❌ Failure Cases
1. **Teacher > Student on FED-RO**: Curriculum learning failed
2. **Large KDEF drop**: Clean face performance degraded
3. **Attention correlates with occlusion**: Copying mask instead of learning

## Advanced Usage

### Compare Multiple Checkpoints

```bash
# Compare different training epochs
for epoch in 20 40 60 80 100; do
    python compare_occluded_performance.py \
        --student_path ../checkpoints/curriculum/student_epoch_${epoch}.pth \
        --output_dir results_epoch_${epoch}
done
```

### Quick Test (Subset)

```bash
# Use only 500 samples per dataset for quick validation
python compare_occluded_performance.py \
    --max_samples 500 \
    --batch_size 64
```

### Custom Datasets

```bash
# Use different clean dataset (e.g., AffectNet)
python compare_occluded_performance.py \
    --kdef_root ../data/AffectNet \
    --output_dir results_affectnet
```

## Troubleshooting

### Issue: Student performs worse on FED-RO

**Possible causes**:
1. Training didn't complete Phase 3/4 properly
2. NaN issues during attention learning
3. Lambda3 too low (attention not emphasized)
4. Beta learning rate too high (attention collapsed)

**Solutions**:
- Check training logs for Phase 3/4 attention metrics
- Verify lambda3 reached 0.1 in Phase 4
- Ensure correlation was in 0.6-0.9 range
- Resume training if stopped early

### Issue: Student worse on KDEF (clean)

**Possible causes**:
1. Overfitting to occluded patterns
2. Attention mechanism hurting clean faces
3. DenseNet weights degraded

**Solutions**:
- Check Phase 1 accuracy (should be high)
- Reduce lambda4 (attention diversity penalty)
- Consider joint training on mixed clean/occluded in Phase 4

### Issue: Attention looks random

**Possible causes**:
1. Beta didn't learn properly (stuck at 0)
2. Lambda3 too low
3. Gradient issues during Phase 3

**Solutions**:
- Check beta value in checkpoints (should be 0.1-0.5 range)
- Verify attention_correlation in logs (0.6-0.9)
- Ensure Phase 3 used lr_beta=1e-5, not frozen

## Files Generated

```
eval_comparison_results/
├── accuracy_comparison.png          # Main result: occluded vs clean
├── confusion_matrices.png           # Detailed FED-RO confusion
├── per_emotion_accuracy.png         # Emotion-level breakdown  
├── attention_visualization.png      # Spatial attention examples
└── performance_comparison_report.txt # Quantitative summary
```

## Scientific Interpretation

This comparison provides **empirical evidence** for:

1. **Curriculum Learning Effectiveness**: Progressive occlusion exposure improves robustness
2. **Spatial Attention Utility**: Learned attention identifies and suppresses occluded regions
3. **No Catastrophic Forgetting**: Student maintains clean face performance
4. **Generalization**: Improvements across multiple emotions, not memorization

**For paper/report**: Use the bar chart and attention visualization as key figures demonstrating the student's superior occlusion handling while maintaining baseline clean face performance.

## Citation

If student outperforms teacher on FED-RO with >5% improvement, this validates:
- Curriculum learning strategy (Section 4.2)
- Spatial attention mechanism (Section 3.2)
- Multi-component distillation loss (Section 3.3)

Include results in your implementation report to show **successful knowledge distillation with occlusion-aware improvements**.

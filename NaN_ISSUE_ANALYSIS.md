# NaN Issue Analysis and Fix - Phase 4 Training

## Issue Description

Training produced NaN values starting at Epoch 41 (Phase 4 beginning):
```
Epoch 41/60 [Phase 4]
  Loss: nan
  Distillation: nan
  Consistency: nan
  Beta: nan
  Attention Reg: nan
  Attention Div: nan
```

## Root Cause Analysis

### Pre-Condition (End of Phase 3, Epoch 40)
- **Attention Correlation: 0.9739** (Target: 0.6-0.9, Safe: <0.95)
- **Beta value: ~0.003** (Started at 0.0)
- **lambda3: 0.1** (Attention regularization weight)

### Two Contributing Factors

#### **Factor 1: Numerical Instability in Entropy Calculation**

**AttentionDiversityLoss** computes binary entropy:
```python
entropy = -(m * log(m) + (1-m) * log(1-m))
```

When attention values approach boundaries (0.0 or 1.0):
- Original epsilon: `1e-8`
- Clamping: `m = clamp(attention, 1e-8, 1 - 1e-8)`
- When `m = 1 - 1e-8`: `(1-m) = 1e-8`
- `log(1e-8) approx -18.4` causes numerical instability
- **Result: NaN propagation**

**Diagnostic Evidence:**
```
TEST 2: Attention Diversity Entropy NaN
All ones       : mean=nan, has_nan=True
Very large     : mean=nan, has_nan=True
```

#### **Factor 2: Beta Gradient Explosion**

**Gradient Magnitudes:**
```
Beta gradient: 1.042171  (HUGE for parameter value 0.003!)
Features gradient norm: 0.006682
```

**The Explosion Chain:**
1. High correlation (0.9739) means attention perfectly matches binary masks
2. Residual formula: `F_attended = F + beta * F * (1 - M_attention)`
3. Strong gradients flow through attention branch
4. Beta gradients amplified by feature magnitudes
5. **Phase 4 change: lr_beta increased from 1e-5 to 5e-5** (5x jump!)
6. Large gradient + higher learning rate = explosion
7. Beta becomes NaN
8. NaN propagates through entire forward pass

**Why Phase 3 Survived:**
- Phase 3: `lr_beta = 1e-5` (conservative)
- Phase 4: `lr_beta = 5e-5` (aggressive, combined with already high correlation)

## Implemented Fixes

### Fix 1: Increase Entropy Epsilon (Numerical Stability)

**File:** `utils/distillation_losses.py`

**Change:**
```python
# Before
def __init__(self, eps=1e-8):

# After  
def __init__(self, eps=1e-6):
```

**Rationale:**
- Provides larger safety margin away from log(0)
- Prevents NaN when attention saturates to 0 or 1
- Negligible impact on entropy values in normal range [0.1, 0.9]

### Fix 2: Reduce Phase 4 Beta Learning Rate

**File:** `train_curriculum.py`

**Change:**
```python
# Phase 4 config
'lr_beta': 1e-5,  # Changed from 5e-5
```

**Rationale:**
- Matches Phase 3 learning rate (no sudden jump)
- Prevents gradient explosion when transitioning to Phase 4
- Beta can still increase, just more gradually
- Safer given that correlation was already high (0.9739)

### Fix 3: Separate Beta Gradient Clipping

**File:** `train_curriculum.py`

**Change:**
```python
# Backward and optimize
loss.backward()

# Clip gradients: general parameters at 1.0, beta separately at 0.1
torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
if config['phase'] >= 3:  # Beta is trainable in Phase 3-4
    torch.nn.utils.clip_grad_norm_([student.attention.beta], max_norm=0.1)

optimizer.step()
```

**Rationale:**
- Beta gradients are consistently ~1.0, much larger than feature gradients (~0.007)
- Separate clipping at 0.1 prevents explosion while allowing learning
- Does not interfere with other parameters (DenseNet, attention conv layers)

### Fix 4: Auto-Adjust Lambda3 on High Correlation

**File:** `train_curriculum.py`

**Change:**
```python
# Auto-adjust lambda3 if correlation too high (>0.95)
if corr > 0.95 and loss_fn.lambda3 > 0.01:
    old_lambda3 = loss_fn.lambda3
    loss_fn.lambda3 *= 0.5  # Reduce by 50%
    print(f"  [WARNING] Correlation too high! Reducing lambda3: {old_lambda3:.4f} to {loss_fn.lambda3:.4f}")
```

**Rationale:**
- Correlation >0.95 indicates attention is copying masks too exactly
- Reduces regularization pressure to allow more flexibility
- Prevents Phase 4 starting with overfitted attention
- Dynamic adjustment based on training behavior

## Expected Behavior After Fixes

### Phase 4 Training Should Now:
1. **No NaN values** - Numerical stability ensured
2. **Gradual beta growth** - Conservative learning rate prevents explosion
3. **Stable gradients** - Beta clipping at 0.1 prevents spikes
4. **Better generalization** - Auto-adjustment prevents mask overfitting

### Monitoring Guidelines:

**Healthy Ranges:**
- Attention Correlation: **0.6 - 0.9** (ideal)
- Beta: Should gradually increase from ~0.003 to ~0.05-0.2 by epoch 60
- Attention Entropy: 0.3 - 0.6 (balanced focus)
- Loss: Should continue decreasing smoothly

**Warning Signs:**
- Correlation >0.95: Auto-adjustment will trigger
- Beta >0.5: May indicate over-reliance on attention
- Entropy <0.2: Attention too peaked (possible collapse)

## Resuming Training

To resume from last good checkpoint (Epoch 40):

```bash
python train_curriculum.py \
    --teacher_path pretrained/FECNet.pt \
    --train_csv data/dataset_pairs_train.csv \
    --test_csv data/dataset_pairs_test.csv \
    --resume_from checkpoints/student_epoch_40.pth
```

The fixes will automatically apply from Epoch 41 onward.

## Technical Insights

### Why High Correlation is Dangerous:

When attention perfectly matches binary masks:
- Attention becomes deterministic (no learning flexibility)
- Gradients concentrate on mask alignment rather than feature quality
- Beta receives strong conflicting signals (suppress vs preserve)
- Small numerical errors amplify during backpropagation

### The Beta Parameter's Role:

```python
F_attended = F + beta * F * (1 - M_attention)
```

- **beta = 0**: Attention has no effect (identity, Phase 1-2 state)
- **beta small (0.01-0.1)**: Gentle attention modulation (healthy)
- **beta medium (0.1-0.3)**: Strong attention influence
- **beta large (>0.5)**: Attention dominates, may suppress too much

### Optimal Training Trajectory:

1. **Phase 1-2**: Beta frozen at 0, learn baseline without attention
2. **Phase 3**: Beta unfreezes, slowly grows 0.0 to 0.01
3. **Phase 4**: Beta continues growing to 0.05-0.2 by epoch 100
4. **Correlation**: Stays in 0.6-0.9 range throughout

## Verification Steps

After fixes, verify:
1. No NaN in any metrics
2. Beta increases smoothly without jumps
3. Correlation stays <0.95 or triggers auto-adjustment
4. Loss decreases steadily
5. Test cosine similarity continues improving

## References

- **OccFECNet.md Section 6.2**: Pearson correlation monitoring (0.6-0.9 target)
- **OccFECNet.md Section 3.5**: Attention diversity loss formulation
- **OccFECNet.md Section 2.2**: Beta parameter initialization and role
- **Debug results**: `debug_nan.py` diagnostic tests

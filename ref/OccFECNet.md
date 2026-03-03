# REFERENCE DOCUMENT: Teacher-Student Knowledge Distillation for Occluded Facial Expression Embedding with Residual Spatial Attention

## 1. Overview

### 1.1 Problem Statement

Train a student FECNet model robust to facial occlusions (face masks, sunglasses, hands) by learning from a pre-trained teacher FECNet trained on clean faces. The student must produce embeddings for occluded faces that match the teacher's embeddings for the same clean faces.

### 1.2 Constraints and Design Decisions

- Datasets: AffectNet, RAF-DB (no access to original FEC dataset)
- Occlusion range: 1% to 40% of face area (beyond 40%, information bottleneck becomes too severe)
- Occlusion types: Face masks, sunglasses, hands (synthetically applied)
- No triplet loss (no triplet annotations available)
- Binary occlusion masks available during training only, not at inference
- Student must handle occluded faces at inference without external mask input


### 1.3 Architecture Overview

- **Teacher**: Pre-trained FECNet (frozen) producing 16-dimensional L2-normalized embeddings
- **Student**: FECNet architecture with added residual spatial attention module inserted after frozen FaceNet layers (7×7×1024) and before DenseNet block
- **Training**: Progressive 4-phase curriculum with identity-initialized attention
- **Inference**: Student operates independently without binary masks

***

## 2. Network Architecture

### 2.1 Base Architecture (FECNet)

Reference: Vemulapalli \& Agarwala, CVPR 2019[^3]

**Input processing**:

- Face alignment: Correct roll rotation, scale to 55-pixel inter-ocular distance
- Resize to 224×224 RGB

**Feature extraction**:

- Frozen FaceNet NN2 layers up to inception block 4e → 7×7×1024 feature maps
- Freezing rationale: FaceNet learned identity-invariant representations from large-scale data; leverage these without retraining

**Expression-specific layers**:

- 1×1 convolution (512 filters) for dimensionality reduction
- DenseNet block (5 layers, growth rate 64) for feature refinement
- 7×7 average pooling → 1×1×(512 + 5×64) = 1×1×832 features
- Fully connected layer (512 hidden units)
- Embedding layer: Linear FC (16 units) + L2 normalization → 16-dimensional unit-norm embedding

**Activations and regularization**:

- Batch normalization after each convolutional layer
- ReLU6 activation throughout
- Dropout for regularization (rate 0.5 during training)


### 2.2 Residual Spatial Attention Module

**Insertion point**: After frozen FaceNet (7×7×1024 features), before 1×1 convolution

**Input**:

- $F \in \mathbb{R}^{7 \times 7 \times 1024}$: FaceNet feature maps
- $M_{\text{binary}} \in \mathbb{R}^{224 \times 224}$: Binary occlusion mask (1=occluded, 0=visible) - **training only**

**Architecture**:

1. **Mask downsampling** (training only):
    - Resize $M_{\text{binary}}$ to 7×7 using average pooling → $M_{\text{down}} \in \mathbb{R}^{7 \times 7}$
    - Average pooling produces soft values (e.g., if 30% of pixels in a patch are occluded, the 7×7 value is 0.3)
    - At inference: $M_{\text{down}}$ is not available; attention operates purely from features
2. **Feature-mask fusion**:
    - Global average pooling on $F$ → $F_{\text{global}} \in \mathbb{R}^{1024}$ (channel-wise global context)
    - Concatenate $M_{\text{down}}$ (if available) as additional channel → $F_{\text{concat}} \in \mathbb{R}^{7 \times 7 \times 1025}$ (training) or $F_{\text{concat}} = F$ (inference)
    - 1×1 convolution (512 filters) + BatchNorm + ReLU → $F_{\text{fused}} \in \mathbb{R}^{7 \times 7 \times 512}$
3. **Attention map generation**:
    - 1×1 convolution (1 filter) → $M_{\text{logits}} \in \mathbb{R}^{7 \times 7 \times 1}$
    - Sigmoid activation → $M_{\text{attention}} \in \mathbb{R}^{7 \times 7}$ with values in[^3]
    - High values indicate uninformative/occluded regions (should be downweighted)
4. **Residual attention connection**:
$F_{\text{attended}} = F + \beta \cdot F \odot (1 - M_{\text{attention}})$

where:
    - $\odot$ is element-wise multiplication (attention broadcast across 1024 channels)
    - $\beta$ is a learnable scalar parameter controlling attention strength
    - When $\beta = 0$ (initialization): $F_{\text{attended}} = F$ (identity/transparent layer)
    - When $\beta > 0$: Attention modulates features additively rather than multiplicatively
    - When $M_{\text{attention}}^{i,j} = 1$: Feature at location $(i,j)$ is downweighted by $\beta$
    - When $M_{\text{attention}}^{i,j} = 0$: Feature at location $(i,j)$ is preserved with small additive boost $\beta F^{i,j}$
5. **Output**: $F_{\text{attended}} \in \mathbb{R}^{7 \times 7 \times 1024}$ fed to 1×1 convolution and subsequent DenseNet layers

**Initialization strategy**:

- 1×1 convolution weights: He initialization with scale 0.01 (small random values)
- 1×1 convolution biases: Initialize to 0 (sigmoid(0) = 0.5, neutral starting point)
- $\beta$: Initialize to 0.0 (attention has zero effect initially, layer is transparent)
- Rationale: Preserves pre-trained FaceNet features at the start, allows progressive specialization

**Inference behavior**:

- At inference, $M_{\text{binary}}$ is not provided
- Feature-mask fusion uses only $F_{\text{concat}} = F$ (no mask channel)
- Attention mechanism learns to detect occlusion patterns from features alone
- During training, the binary mask guides learning; at inference, the learned attention generalizes

***

## 3. Loss Function Formulation

### 3.1 Multi-Component Loss

$L_{\text{total}} = \lambda_1 L_{\text{distill}} + \lambda_2 L_{\text{consistency}} + \lambda_3 L_{\text{attention-reg}} - \lambda_4 L_{\text{attention-diversity}}$

All loss components are computed per mini-batch and averaged.

### 3.2 Component 1: Distillation Loss

**Objective**: Student embeddings for occluded faces should match teacher embeddings for clean faces

**Formulation** (Cosine Embedding Loss):

$L_{\text{distill}} = 1 - \frac{e_{\text{teacher}} \cdot e_{\text{student}}}{\|e_{\text{teacher}}\|_2 \|e_{\text{student}}\|_2}$

Since both embeddings are L2-normalized (by FECNet design), this simplifies to:

$L_{\text{distill}} = 1 - (e_{\text{teacher}} \cdot e_{\text{student}})$

**Interpretation**:

- Value 0: Perfect alignment (embeddings point in same direction)
- Value 1: Orthogonal embeddings (no similarity)
- Value 2: Opposite embeddings (maximum dissimilarity)

**Input pairs**:

- Teacher input: $I_{\text{clean}}$ (original clean face)
- Student input: $I_{\text{occluded}}$ (same face with synthetic occlusion)
- Constraint: Embeddings should be nearly identical despite different inputs

**Gradient properties**:

- Cosine loss has well-behaved gradients for unit-norm embeddings
- Focuses on angular alignment rather than magnitude
- More interpretable than L2 loss


### 3.3 Component 2: Consistency Loss

**Objective**: Student must maintain performance on clean faces (prevent catastrophic forgetting)

**Formulation** (Cosine Embedding Loss):

$L_{\text{consistency}} = 1 - (e_{\text{teacher}}(I_{\text{clean}}) \cdot e_{\text{student}}(I_{\text{clean}}))$

**Input pairs**:

- Both teacher and student receive the same clean face
- Ensures student doesn't diverge from teacher on clean data

**Rationale**:

- Without this term, student overfits to occluded data and degrades on clean faces
- Maintains backward compatibility with teacher's clean-face performance
- Cosine loss is theoretically sound here for same reasons as distillation loss

**Batch composition**:

- Each mini-batch contains 50% occluded pairs (for distillation) and 50% clean pairs (for consistency)
- Ensures balanced gradient updates for both objectives


### 3.4 Component 3: Attention Regularization

**Objective**: Penalize attention mechanism if it attends to (assigns low attention values to) occluded regions

**Formulation**:

$L_{\text{attention-reg}} = \frac{1}{|S|} \sum_{(i,j) \in S} (1 - M_{\text{attention}}^{i,j}) \cdot M_{\text{down}}^{i,j}$

where $S = \{(i,j) : 1 \leq i,j \leq 7\}$ is the set of spatial locations in the 7×7 feature map.

**Interpretation**:

- When location $(i,j)$ is occluded ($M_{\text{down}}^{i,j} = 1$):
    - If attention correctly identifies it ($M_{\text{attention}}^{i,j} = 1$): Loss contribution is $(1-1) \cdot 1 = 0$ (good)
    - If attention incorrectly keeps it ($M_{\text{attention}}^{i,j} = 0$): Loss contribution is $(1-0) \cdot 1 = 1$ (penalized)
- When location $(i,j)$ is visible ($M_{\text{down}}^{i,j} = 0$):
    - Loss contribution is always 0 regardless of $M_{\text{attention}}^{i,j}$
    - Allows model to freely learn which visible regions are informative

**Design rationale**:

- Only penalizes attention to occluded regions, doesn't constrain attention on visible regions
- More flexible than binary cross-entropy (which would force attention to exactly match mask)
- Enables model to learn that some visible regions (e.g., background) are also uninformative

**Training-only loss**:

- Requires $M_{\text{down}}$ which is only available during training
- At inference, attention operates independently without this loss


### 3.5 Component 4: Attention Diversity Regularization

**Objective**: Prevent attention collapse (all locations assigned same attention value)

**Formulation** (Entropy Regularization):

$L_{\text{attention-diversity}} = -\frac{1}{|S|} \sum_{(i,j) \in S} \left[M_{\text{attention}}^{i,j} \log M_{\text{attention}}^{i,j} + (1 - M_{\text{attention}}^{i,j}) \log (1 - M_{\text{attention}}^{i,j})\right]$

**Note**: This term has a **negative sign** in the total loss (we subtract it, effectively maximizing entropy)

**Interpretation**:

- Entropy is maximized when $M_{\text{attention}}^{i,j} = 0.5$ for all locations (uniform distribution)
- Entropy is minimized when all values are 0 or 1 (deterministic)
- Encourages diverse attention patterns: some regions high attention (downweighted), others low attention (preserved)

**Prevents collapse modes**:

- Without this term, attention might collapse to $M_{\text{attention}} \approx 1$ everywhere (downweight all features, rely on bias terms)
- Or collapse to $M_{\text{attention}} \approx 0$ everywhere (ignore attention, become transparent)
- Entropy regularization ensures spatial variation in attention

**Balancing with other losses**:

- Attention regularization ($L_3$) pushes attention toward 1 at occluded locations
- Diversity regularization ($L_4$) pulls attention toward 0.5 everywhere
- Together they balance: occluded regions have high attention (~0.8-0.9), visible regions have lower but non-zero attention (~0.2-0.4)


### 3.6 Hyperparameter Values and Curriculum

**Recommended initial values**:

- $\lambda_1 = 1.0$ (distillation is primary objective, baseline scale)
- $\lambda_2 = 0.5$ (consistency is important but secondary to occlusion robustness)
- $\lambda_3 = 0.0$ initially, increase progressively (see curriculum below)
- $\lambda_4 = 0.01$ (weak regularization to prevent collapse)

**Curriculum for $\lambda_3$** (Attention Regularization):


| Training Phase | Epochs | $\lambda_3$ Value | Rationale |
| :-- | :-- | :-- | :-- |
| Phase 1 | 1-10 | 0.0 | Attention is inactive ($\beta = 0$), no need to regularize |
| Phase 2 | 11-20 | 0.0 | Attention begins activating but learning basic occlusion patterns, don't constrain yet |
| Phase 3 | 21-40 | 0.05 → 0.1 (linear increase) | Attention is active, progressively guide it toward mask-aligned behavior |
| Phase 4 | 41-60 | 0.1 | Full regularization, attention mature |

**Hyperparameter tuning guidance**:

- If attention ignores binary masks (low correlation between $M_{\text{attention}}$ and $M_{\text{down}}$): Increase $\lambda_3$
- If attention exactly copies binary masks (correlation > 0.95): Decrease $\lambda_3$
- If attention collapses to uniform values (low variance across spatial locations): Increase $\lambda_4$
- If distillation loss plateaus early: Increase $\lambda_1$ or decrease $\lambda_2, \lambda_3$

***

## 4. Training Procedure

### 4.1 Data Preparation

**Datasets**:

- AffectNet: 287,651 training images, 3,500 validation images across 8 emotion categories
- RAF-DB: 12,271 training images, 3,068 test images across 7 emotion categories
- Combine both datasets for training (ignore emotion labels since no triplet loss)

**Synthetic occlusion generation**:

For each clean face image:

1. **Face masks** (covers mouth and nose):
    - Detect facial landmarks (use dlib or MediaPipe)
    - Generate polygon mask covering nose tip to chin
    - Texture: Random solid colors (black, white, blue, medical green) or realistic surgical mask patterns
    - Occlusion coverage: 20-40% of face
2. **Sunglasses** (covers eyes and eyebrows):
    - Detect eye landmarks
    - Generate elliptical masks covering both eyes with slight margin
    - Texture: Black (dark sunglasses) or mirrored silver
    - Occlusion coverage: 10-25% of face
3. **Hands** (covers random face regions):
    - Random rectangular or irregular polygon shapes
    - Position: Random placement covering 10-40% of face
    - Texture: Skin-tone colors sampled from face regions
    - Variations: Single hand (partial), two hands (extended coverage)
4. **Occlusion diversity**:
    - Each training image generates 3 occluded versions (one per occlusion type)
    - Augmentation: Random variations in occlusion position, size, orientation
    - Progressive severity: Start with 1% coverage (minimal occlusion), gradually increase to 40% (severe occlusion)

**Binary mask generation**:

- For each occluded image, generate corresponding binary mask $M_{\text{binary}} \in \{0,1\}^{224 \times 224}$
- Value 1 at occluded pixels, 0 at visible pixels
- Store mask-image pairs for training


### 4.2 Progressive Training Curriculum (4 Phases)

#### Phase 1: Baseline Distillation (Epochs 1-10)

**Goal**: Establish baseline - student learns to match teacher on clean faces without attention

**Configuration**:

- Student architecture: Frozen FaceNet + DenseNet (no attention module yet, or attention inactive)
- Input: Clean faces only (no occlusion)
- Loss: $L = \lambda_1 L_{\text{distill}}$ only
    - Teacher input: $I_{\text{clean}}$
    - Student input: $I_{\text{clean}}$ (same image)
- Hyperparameters: $\lambda_1 = 1.0$, all others zero
- Optimizer: Adam, learning rate 5e-4
- Batch size: 90

**Expected outcome**:

- Distillation loss should approach near-zero (< 0.01)
- Student embeddings nearly identical to teacher embeddings on clean faces
- Validates student can replicate teacher performance

**Checkpoint validation**:

- Compute cosine similarity between teacher and student embeddings on validation set
- Target: Mean cosine similarity > 0.99


#### Phase 2: Introduce Occlusion + Identity Attention (Epochs 11-20)

**Goal**: Expose student to occlusion while keeping attention transparent

**Configuration**:

- Add attention module with identity initialization ($\beta = 0$, frozen)
- Input: 50% occluded faces, 50% clean faces
- Occlusion severity: Progressive from 1% to 20% coverage (curriculum within phase)
- Loss: $L = \lambda_1 L_{\text{distill}} + \lambda_2 L_{\text{consistency}}$
    - Distillation loss on occluded pairs
    - Consistency loss on clean pairs
- Hyperparameters: $\lambda_1 = 1.0$, $\lambda_2 = 0.5$, $\lambda_3 = 0$, $\lambda_4 = 0$
- Optimizer: Adam, learning rate 5e-4
- Batch size: 90 (45 occluded, 45 clean)

**Occlusion curriculum within phase**:

- Epochs 11-13: 1-5% occlusion (minimal)
- Epochs 14-16: 5-10% occlusion
- Epochs 17-20: 10-20% occlusion

**Expected outcome**:

- Distillation loss increases (student struggles with occlusion) but should converge
- Consistency loss remains low (student maintains clean-face performance)
- Attention module present but inactive (gradients flow through but output unchanged)

**Checkpoint validation**:

- Mean attention values should be ~0.5 (neutral, not collapsed)
- Verify $\beta \approx 0$ (attention frozen)


#### Phase 3: Activate Attention (Epochs 21-40)

**Goal**: Unfreeze attention, teach it to detect and downweight occluded regions

**Configuration**:

- Unfreeze $\beta$: Initialize to 0.0, make learnable with learning rate 1e-5
- Unfreeze attention convolution layers with learning rate 1e-4
- Occlusion severity: Progressive from 20% to 40% coverage
- Loss: Full multi-component loss
$L = \lambda_1 L_{\text{distill}} + \lambda_2 L_{\text{consistency}} + \lambda_3 L_{\text{attention-reg}} - \lambda_4 L_{\text{attention-diversity}}$
- Hyperparameters:
    - $\lambda_1 = 1.0$
    - $\lambda_2 = 0.5$
    - $\lambda_3$: Progressive from 0.0 → 0.05 (epochs 21-30) → 0.1 (epochs 31-40)
    - $\lambda_4 = 0.01$
- Optimizer: Adam, learning rate 5e-4 for DenseNet, 1e-4 for attention layers, 1e-5 for $\beta$
- Batch size: 90 (45 occluded, 45 clean)

**Occlusion curriculum within phase**:

- Epochs 21-25: 20-25% occlusion
- Epochs 26-30: 25-30% occlusion
- Epochs 31-35: 30-35% occlusion
- Epochs 36-40: 35-40% occlusion

**Expected outcome**:

- Distillation loss should decrease as attention learns to filter occluded features
- $\beta$ should increase gradually from 0.0 toward 0.5-0.8
- Attention maps should correlate with binary masks (Pearson correlation > 0.6)
- Attention should show spatial diversity (not uniform across all locations)

**Checkpoint validation**:

- Visualize attention maps: Should highlight occluded regions (high values) and preserve visible regions (low values)
- Compute correlation between $M_{\text{attention}}$ and $M_{\text{down}}$: Target > 0.6, < 0.95
- Monitor $\beta$ value: Should be in range [0.3, 0.8]


#### Phase 4: Full Training (Epochs 41-60)

**Goal**: Finalize training with full occlusion range and stable hyperparameters

**Configuration**:

- All parameters unfrozen with finalized learning rates
- Occlusion severity: Uniform random sampling from 10% to 40% (no longer progressive)
- Loss: Same as Phase 3
- Hyperparameters: $\lambda_1 = 1.0$, $\lambda_2 = 0.5$, $\lambda_3 = 0.1$, $\lambda_4 = 0.01$
- Optimizer: Adam, learning rate 5e-4 for DenseNet, 1e-4 for attention layers, 5e-5 for $\beta$
- Batch size: 90 (45 occluded, 45 clean)
- Optional: Introduce mask noise (dilate/erode binary masks by 1-2 pixels) to simulate imperfect mask detection

**Expected outcome**:

- All losses stabilize and converge
- Distillation loss should be significantly lower than Phase 2 (attention successfully filters occlusion)
- Consistency loss remains low (clean-face performance preserved)
- Attention-regularization loss low (attention aligns with masks)
- Attention-diversity loss moderate (attention is diverse, not collapsed)

**Checkpoint validation**:

- Evaluate on held-out test set with novel occlusion patterns
- Compute mean cosine similarity between teacher (clean) and student (occluded) embeddings
- Target: > 0.85 for 10-20% occlusion, > 0.75 for 30-40% occlusion
- Visualize attention maps on test samples to verify generalization


### 4.3 Optimizer and Regularization

**Optimizer**: Adam

- Beta1: 0.9
- Beta2: 0.999
- Epsilon: 1e-8
- Weight decay: 1e-4 (L2 regularization on all trainable parameters)

**Learning rate schedule**:

- DenseNet layers: Start 5e-4, cosine annealing to 1e-5 over 60 epochs
- Attention layers: Start 1e-4, cosine annealing to 1e-6 over 60 epochs
- $\beta$ parameter: Start 1e-5 (Phase 3), increase to 5e-5 (Phase 4), no annealing

**Dropout**:

- Apply dropout 0.5 after DenseNet fully connected layer
- No dropout in attention module (causes instability in attention maps)

**Batch normalization**:

- Apply after all convolutional layers (frozen FaceNet already has BN)
- Use batch statistics during training, running statistics during inference

**Gradient clipping**:

- Clip gradients by global norm with threshold 1.0
- Prevents exploding gradients in attention layers during early training


### 4.4 Data Augmentation

**Applied to all images** (both clean and occluded):

- Random horizontal flip (probability 0.5)
- Random brightness adjustment (±10%)
- Random contrast adjustment (±10%)
- Random rotation (±5 degrees)
- No aggressive augmentation (would interfere with occlusion patterns)

**Not applied**:

- Random cropping (would misalign faces, breaking FaceNet assumptions)
- Color jittering (faces should maintain natural skin tones)
- Cutout/random erasing (conflicts with synthetic occlusion)

***

## 5. Inference Procedure

### 5.1 Inference Without Binary Masks

**Key property**: At inference, binary occlusion masks $M_{\text{binary}}$ are not available or provided

**Modified attention module behavior**:

- Feature-mask fusion step uses only $F_{\text{concat}} = F$ (no mask channel concatenated)
- 1×1 convolution operates on 7×7×1024 features only
- Attention map $M_{\text{attention}}$ generated purely from learned feature patterns
- Residual connection: $F_{\text{attended}} = F + \beta \cdot F \odot (1 - M_{\text{attention}})$

**How attention generalizes without masks**:

- During training, attention learned to associate certain feature patterns (blur, edges, texture discontinuities) with occlusion
- These patterns are present in FaceNet features even without explicit mask input
- Attention mechanism acts as an **occlusion detector**: Recognizes occluded regions from features and downweights them
- Analogous to self-supervised learning: Pretext task (mask prediction) provides training signal, but learned representation generalizes

**Graceful degradation**:

- Residual connection ensures that if attention fails (predicts uniform values), the model falls back to using full features
- Unlike hard masking (which would zero features if mask detection fails), residual formulation preserves information


### 5.2 Inference Pipeline

**Input**: Single face image $I_{\text{test}} \in \mathbb{R}^{224 \times 224 \times 3}$ (potentially occluded)

**Steps**:

1. **Preprocessing**:
    - Face alignment (same as training): Detect landmarks, correct rotation, scale to 55-pixel inter-ocular distance
    - Resize to 224×224
2. **Teacher embedding** (optional, for comparison):
    - If clean version of face is available: $e_{\text{teacher}} = \text{FECNet}_{\text{teacher}}(I_{\text{clean}})$
    - Typically not available at inference (occlusion is unknown)
3. **Student inference**:
    - Forward pass through frozen FaceNet: $F = \text{FaceNet}(I_{\text{test}}) \in \mathbb{R}^{7 \times 7 \times 1024}$
    - Attention module (without mask input):
        - $F_{\text{global}} = \text{GlobalAvgPool}(F) \in \mathbb{R}^{1024}$
        - $F_{\text{fused}} = \text{Conv1x1}(F) \in \mathbb{R}^{7 \times 7 \times 512}$
        - $M_{\text{attention}} = \sigma(\text{Conv1x1}(F_{\text{fused}})) \in \mathbb{R}^{7 \times 7}$
        - $F_{\text{attended}} = F + \beta \cdot F \odot (1 - M_{\text{attention}})$
    - DenseNet block + FC layers + L2 normalization: $e_{\text{student}} = \text{DenseNet}(F_{\text{attended}}) \in \mathbb{R}^{16}$
4. **Output**: 16-dimensional L2-normalized embedding $e_{\text{student}}$

**Downstream tasks**:

- Expression similarity: Compute cosine similarity between two embeddings
- Expression retrieval: Find K-nearest neighbors in embedding space
- Expression clustering: Apply hierarchical clustering on embeddings


### 5.3 Attention Visualization (Optional)

For interpretability, visualize attention maps at inference:

1. Extract $M_{\text{attention}} \in \mathbb{R}^{7 \times 7}$ from attention module
2. Upsample to 224×224 using bilinear interpolation
3. Overlay on original image as heatmap (red = high attention = downweighted regions)
4. Verify attention highlights occluded areas (e.g., face mask, sunglasses)

This provides transparency into model's decision-making: Which regions did it consider uninformative?

***

## 6. Evaluation Metrics

### 6.1 Primary Metrics (Quantitative)

**Cosine Similarity on Paired Clean-Occluded Faces**:

- Dataset: Hold-out test set with paired images (same person, same expression, one clean, one occluded)
- Metric: Mean cosine similarity $\frac{1}{N}\sum_{i=1}^{N} (e_{\text{teacher}}(I_{\text{clean}}^i) \cdot e_{\text{student}}(I_{\text{occluded}}^i))$
- Target: > 0.85 for mild occlusion (10-20%), > 0.75 for severe occlusion (30-40%)
- Baseline comparison: Teacher on clean vs clean (should be ~0.99), teacher on clean vs occluded without student (degrades to ~0.5-0.6)

**Breakdown by Occlusion Type**:

- Report mean cosine similarity separately for face masks, sunglasses, hands
- Identifies which occlusion types are handled better/worse

**Breakdown by Occlusion Severity**:

- Bin occlusions by coverage: [1-10%], [10-20%], [20-30%], [30-40%]
- Plot cosine similarity vs occlusion percentage (should decrease gradually, not cliff-drop)


### 6.2 Attention Quality Metrics

**Attention-Mask Correlation** (training set only, where masks available):

- Pearson correlation between $M_{\text{attention}}$ and $M_{\text{down}}$
- Target: 0.6-0.9 (high correlation but not perfect, indicating learned generalization)
- Too low (< 0.5): Attention ignores masks
- Too high (> 0.95): Attention overfits to masks, may not generalize

**Attention Sparsity**:

- Mean attention value: $\bar{M} = \frac{1}{49}\sum_{i,j} M_{\text{attention}}^{i,j}$
- Target: 0.3-0.5 for 30-40% occlusion (should downweight roughly the same percentage as occluded)
- Value ~0.5 everywhere indicates collapse (uninformative attention)

**Attention Entropy**:

- Measure spatial diversity: $H = -\sum_{i,j} p_{i,j} \log p_{i,j}$ where $p_{i,j} = \frac{M_{\text{attention}}^{i,j}}{\sum M_{\text{attention}}}$
- Target: High entropy (diverse attention across locations)
- Low entropy indicates attention focuses on few locations (may miss distributed occlusions)


### 6.3 Consistency Check (Clean Face Performance)

**Teacher vs Student on Clean Faces**:

- Dataset: Clean validation faces (no occlusion)
- Metric: Mean cosine similarity $\frac{1}{N}\sum_{i=1}^{N} (e_{\text{teacher}}(I_{\text{clean}}^i) \cdot e_{\text{student}}(I_{\text{clean}}^i))$
- Target: > 0.95 (student maintains teacher performance without degradation)
- Critical: Ensures student didn't catastrophically forget clean-face representation


### 6.4 Qualitative Evaluation

**Attention Map Visualization**:

- Select 20-30 test images with diverse occlusion patterns
- Generate attention heatmaps overlaid on original images
- Manually verify attention highlights occluded regions (face masks, sunglasses, hands)
- Check for false positives (attention highlights visible regions) and false negatives (misses occluded regions)

**Embedding Space Visualization**:

- Use t-SNE or UMAP to project embeddings to 2D
- Plot clean faces (blue), occluded faces (red), connected by lines for same-identity pairs
- Target: Same-identity pairs should cluster closely, indicating embedding similarity despite occlusion

**Failure Case Analysis**:

- Identify test samples with low cosine similarity (< 0.6)
- Analyze patterns: Extreme occlusion (> 40%)? Novel occlusion types? Alignment failures?
- Provides insights for future improvement

***

## 7. Expected Challenges and Mitigation Strategies

### 7.1 Challenge: Attention Overfitting to Training Masks

**Symptom**: Attention maps at inference don't generalize to novel occlusion patterns; performance degrades on real-world occlusions not seen during training

**Mitigation**:

- Introduce mask noise during training: Randomly dilate/erode binary masks by 1-3 pixels (simulates imperfect mask detection)
- Add random mask perturbations: Remove 10-20% of mask pixels (simulate mask detection false negatives)
- Use diverse synthetic occlusion patterns: Vary mask shapes, sizes, positions, textures
- Lower $\lambda_3$ (attention regularization weight) to allow more flexibility


### 7.2 Challenge: Attention Collapse

**Symptom**: All attention values converge to same value (e.g., 0.5 everywhere); attention provides no information

**Mitigation**:

- Increase $\lambda_4$ (attention diversity regularization) to 0.05-0.1
- Monitor attention entropy during training; if entropy drops below threshold, increase $\lambda_4$
- Verify $\beta$ is increasing (if $\beta$ stays near 0, attention is inactive)
- Reduce other loss weights ($\lambda_1, \lambda_2$) to give attention loss more influence


### 7.3 Challenge: Distillation-Attention Trade-off

**Symptom**: Lowering distillation loss increases attention-regularization loss (and vice versa); losses compete rather than cooperate

**Mitigation**:

- Tune $\lambda_3$ carefully: Start very low (0.01) and increase gradually
- If conflict persists, prioritize distillation loss (the primary objective); attention is a means to achieve it
- Consider alternative formulation: Replace $L_{\text{attention-reg}}$ with softer version that allows attention to deviate from mask if it improves distillation


### 7.4 Challenge: Clean Face Performance Degradation

**Symptom**: Consistency loss increases over training; student performs worse on clean faces than teacher

**Mitigation**:

- Increase $\lambda_2$ (consistency weight) from 0.5 to 1.0
- Ensure batch composition is exactly 50% clean, 50% occluded (not skewed)
- Check if attention is active on clean faces (should be mostly low values, preserving features)
- Consider adding explicit regularization: $\|M_{\text{attention}}\|_1$ penalty on clean faces (encourage attention to be zero when no occlusion present)


### 7.5 Challenge: FaceNet Feature Misalignment

**Symptom**: Frozen FaceNet produces poor features for occluded faces; attention cannot compensate

**Mitigation**:

- Verify face alignment preprocessing is robust to occlusion (landmarks may be occluded)
- Consider using occlusion-aware landmark detection (detect only visible landmarks)
- If alignment consistently fails, consider fine-tuning FaceNet (unfreeze last few layers) in Phase 4
- Alternative: Use different pre-trained backbone more robust to occlusion (e.g., ArcFace trained with masked faces)

***

## 8. Implementation Checklist

### 8.1 Data Preparation

- [ ] Download and preprocess AffectNet and RAF-DB datasets
- [ ] Implement synthetic occlusion generation (face masks, sunglasses, hands)
- [ ] Generate binary masks for all occluded images
- [ ] Split into train/validation/test sets (80/10/10)
- [ ] Verify occlusion coverage distribution (should span 1-40%)


### 8.2 Model Architecture

- [ ] Load pre-trained FaceNet NN2 checkpoint
- [ ] Freeze FaceNet layers up to inception block 4e
- [ ] Implement residual spatial attention module with identity initialization
- [ ] Verify forward pass produces correct tensor shapes (7×7×1024 → 16D embedding)
- [ ] Test attention module in isolation (input: features + mask, output: attended features)


### 8.3 Loss Functions

- [ ] Implement cosine embedding loss for distillation
- [ ] Implement cosine embedding loss for consistency
- [ ] Implement attention regularization loss (mask-guided)
- [ ] Implement attention diversity loss (entropy)
- [ ] Combine into multi-component loss with configurable weights


### 8.4 Training Loop

- [ ] Implement 4-phase curriculum with progressive occlusion severity
- [ ] Implement $\lambda_3$ curriculum (0 → 0.05 → 0.1)
- [ ] Implement $\beta$ unfreezing schedule (frozen → learnable)
- [ ] Configure separate learning rates for DenseNet, attention layers, $\beta$
- [ ] Add gradient clipping (norm threshold 1.0)
- [ ] Implement batch composition (50% occluded, 50% clean)


### 8.5 Evaluation

- [ ] Implement cosine similarity evaluation on paired clean-occluded faces
- [ ] Implement attention-mask correlation metric
- [ ] Implement attention sparsity and entropy metrics
- [ ] Implement consistency check (teacher vs student on clean faces)
- [ ] Visualize attention maps on test samples


### 8.6 Monitoring and Debugging

- [ ] Log all loss components separately (distillation, consistency, attention-reg, attention-diversity)
- [ ] Log $\beta$ value over training (should increase from 0)
- [ ] Log attention statistics (mean, std, entropy) per epoch
- [ ] Visualize attention maps on validation set every 5 epochs
- [ ] Save checkpoints at end of each phase
- [ ] Implement early stopping based on validation cosine similarity

***

## 9. Recommended Hyperparameters (Summary)

| Parameter | Value | Description |
| :-- | :-- | :-- |
| Embedding dimension | 16 | Fixed by FECNet architecture |
| Batch size | 90 | 45 occluded + 45 clean |
| Total epochs | 60 | Across 4 phases |
| Optimizer | Adam | Beta1=0.9, Beta2=0.999 |
| Weight decay | 1e-4 | L2 regularization |
| Dropout | 0.5 | After DenseNet FC layer |
| Gradient clip norm | 1.0 | Prevents exploding gradients |
| $\lambda_1$ (distillation) | 1.0 | Primary objective |
| $\lambda_2$ (consistency) | 0.5 | Secondary objective |
| $\lambda_3$ (attention-reg) | 0 → 0.05 → 0.1 | Progressive curriculum |
| $\lambda_4$ (attention-diversity) | 0.01 | Weak regularization |
| $\beta$ initialization | 0.0 | Identity/transparent attention |
| DenseNet learning rate | 5e-4 → 1e-5 | Cosine annealing |
| Attention learning rate | 1e-4 → 1e-6 | Cosine annealing |
| $\beta$ learning rate | 1e-5 → 5e-5 | Increase over phases |


***

## 10. References and Theoretical Foundations

### 10.1 Base Architecture

- **FECNet**: Vemulapalli \& Agarwala, "A Compact Embedding for Facial Expression Similarity", CVPR 2019[^3]
- **FaceNet**: Schroff et al., "FaceNet: A Unified Embedding for Face Recognition and Clustering", CVPR 2015
- **DenseNet**: Huang et al., "Densely Connected Convolutional Networks", CVPR 2017


### 10.2 Knowledge Distillation for Occlusion

- Teacher-student training for facial expression under occlusion[^8][^9]
- Information gap-based knowledge distillation for occluded faces[^10]


### 10.3 Attention Mechanisms

- Residual attention connections[^7][^6]
- Self-supervised attention learning without masks[^4][^5]
- Spatial attention for occlusion-robust recognition[^11][^12]


### 10.4 Loss Functions

- Cosine embedding loss for representation learning[^13][^2][^1]
- Consistency regularization in semi-supervised learning[^14][^15]
- Feature distance loss for invariant representations[^14]


### 10.5 Progressive Training

- Progressive layer unfreezing for knowledge distillation[^16][^17]
- Identity mapping initialization in residual networks[^18][^19][^6]

***

## 11. Practical Notes

### 11.1 Computational Requirements

- **Training time**: ~24-36 hours on single NVIDIA V100 GPU (depends on dataset size)
- **Memory**: ~12-16 GB GPU memory for batch size 90
- **Storage**: ~50 GB for datasets + synthetic occlusions + checkpoints


### 11.2 Inference Efficiency

- **Speed**: ~5-10 ms per image on V100 GPU (batch size 1)
- **Memory**: ~2 GB GPU memory for inference
- **Bottleneck**: Frozen FaceNet (most compute-intensive component)


### 11.3 Recommended Tools

- **Framework**: PyTorch 2.0+ (for improved performance)
- **Face alignment**: dlib or MediaPipe
- **Visualization**: Matplotlib, OpenCV for attention heatmaps
- **Experiment tracking**: Weights \& Biases or TensorBoard

***

## END OF REFERENCE DOCUMENT

<span style="display:none">[^20][^21][^22]</span>

<div align="center">⁂</div>

[^1]: https://www.emergentmind.com/topics/cosine-similarity-regularization

[^2]: https://research.netflix.com/publication/is-cosine-similarity-of-embeddings-really-about-similarity

[^3]: Vemulapalli_A_Compact_Embedding_for_Facial_Expression_Similarity_CVPR_2019_paper.pdf

[^4]: https://arxiv.org/pdf/2305.15684.pdf

[^5]: https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Self-Supervised_Equivariant_Attention_Mechanism_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2020_paper.pdf

[^6]: https://arxiv.org/abs/1603.05027

[^7]: https://home.ttic.edu/~savarese/savarese_files/Residual_Gates.pdf

[^8]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8693600/

[^9]: https://arxiv.org/abs/2111.10561

[^10]: https://www.sciencedirect.com/science/article/abs/pii/S0262885624004700

[^11]: https://openaccess.thecvf.com/content/CVPR2021W/CVMI/papers/Zhang_A_Joint_Spatial_and_Magnification_Based_Attention_Framework_for_Large_CVPRW_2021_paper.pdf

[^12]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12349548/

[^13]: https://docs.pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html

[^14]: https://arxiv.org/pdf/2112.05825.pdf

[^15]: https://pure.mpg.de/rest/items/item_3487396/component/file_3487397/content

[^16]: https://www.themoonlight.io/en/review/progressive-knowledge-distillation-of-stable-diffusion-xl-using-layer-level-loss

[^17]: https://www.nature.com/articles/s41598-025-91152-3

[^18]: https://d2l.ai/chapter_convolutional-modern/resnet.html

[^19]: https://openreview.net/pdf?id=EYCm0AFjaSS

[^20]: https://www.sciencedirect.com/science/article/pii/S136184151930057X

[^21]: https://penghao-bdsc.github.io/papers/CoSENT_TASLP2024.pdf

[^22]: https://learnopencv.com/attention-mechanism-in-transformer-neural-networks/


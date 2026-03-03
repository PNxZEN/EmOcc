"""
Multi-Component Loss Functions for Knowledge Distillation
Reference: OccFECNet.md - Section 3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineEmbeddingLoss(nn.Module):
    """
    Cosine Embedding Loss for L2-normalized embeddings
    Reference: OccFECNet.md Section 3.2 and 3.3
    
    For L2-normalized embeddings, cosine similarity = dot product
    Loss = 1 - cos(e1, e2) = 1 - (e1 · e2)
    
    Values:
    - 0: Perfect alignment (same direction)
    - 1: Orthogonal (no similarity)
    - 2: Opposite direction (maximum dissimilarity)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, embedding1, embedding2):
        """
        Args:
            embedding1: [B, D] - First embeddings (L2-normalized)
            embedding2: [B, D] - Second embeddings (L2-normalized)
        
        Returns:
            loss: Scalar - Mean cosine distance
        """
        # Cosine similarity (dot product for L2-normalized vectors)
        cos_sim = (embedding1 * embedding2).sum(dim=1)  # [B]
        
        # Cosine distance
        loss = 1 - cos_sim  # [B]
        
        return loss.mean()


class AttentionRegularizationLoss(nn.Module):
    """
    Attention Regularization Loss (mask-guided)
    Reference: OccFECNet.md Section 3.4
    
    Penalizes attention if it attends to occluded regions.
    Loss = mean((1 - M_attention) * M_down)
    
    Where:
    - M_attention: Predicted attention map [0, 1]
    - M_down: Binary mask downsampled to 5x5 (1=occluded, 0=visible)
      Note: Using 5x5 to match InceptionResnetV1 output (vs 7x7 for FaceNet NN2 in paper)
    
    Training only (requires binary mask).
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, attention_map, binary_mask_down):
        """
        Args:
            attention_map: [B, 5, 5] - Predicted attention (high = occluded)
            binary_mask_down: [B, 5, 5] - Downsampled binary mask (1=occluded, 0=visible)
        
        Returns:
            loss: Scalar - Attention regularization loss
        """
        # Penalize low attention values at occluded locations
        # (1 - M_attention) is high when attention is low (should attend)
        # Multiplied by M_down (1 at occluded locations)
        loss = (1 - attention_map) * binary_mask_down  # [B, 7, 7]
        
        return loss.mean()


class AttentionDiversityLoss(nn.Module):
    """
    Attention Diversity Loss (entropy regularization)
    Reference: OccFECNet.md Section 3.5
    
    Prevents attention collapse by maximizing entropy.
    Loss = -mean(M*log(M) + (1-M)*log(1-M))
    
    Note: This term has NEGATIVE sign in total loss (we subtract it).
    """
    
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, attention_map):
        """
        Args:
            attention_map: [B, 5, 5] - Predicted attention values in [0, 1]
        
        Returns:
            loss: Scalar - Negative entropy (to be subtracted in total loss)
        """
        # Clamp to avoid log(0)
        m = torch.clamp(attention_map, self.eps, 1 - self.eps)
        
        # Binary entropy: -[p*log(p) + (1-p)*log(1-p)]
        entropy = -(m * torch.log(m) + (1 - m) * torch.log(1 - m))  # [B, 7, 7]
        
        # Return mean entropy (will be subtracted in total loss)
        return entropy.mean()


class MultiComponentLoss(nn.Module):
    """
    Multi-Component Loss for Teacher-Student Distillation
    Reference: OccFECNet.md Section 3.1
    
    L_total = lambda1*L_distill + lambda2*L_consistency + lambda3*L_attention_reg - lambda4*L_attention_diversity
    
    Components:
    1. Distillation: Student(occluded) should match Teacher(clean)
    2. Consistency: Student(clean) should match Teacher(clean)
    3. Attention Reg: Guide attention to detect occluded regions
    4. Attention Diversity: Prevent attention collapse
    """
    
    def __init__(self, lambda1=1.0, lambda2=0.5, lambda3=0.0, lambda4=0.01):
        """
        Args:
            lambda1: Weight for distillation loss (default: 1.0)
            lambda2: Weight for consistency loss (default: 0.5)
            lambda3: Weight for attention regularization (default: 0.0, progressive)
            lambda4: Weight for attention diversity (default: 0.01)
        """
        super().__init__()
        
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        
        # Loss components
        self.cosine_loss = CosineEmbeddingLoss()
        self.attention_reg = AttentionRegularizationLoss()
        self.attention_div = AttentionDiversityLoss()
    
    def forward(self, 
                teacher_embed_clean, 
                student_embed_occluded, 
                student_embed_clean,
                attention_map_occluded,
                attention_map_clean,
                binary_mask_down):
        """
        Compute multi-component loss
        
        Args:
            teacher_embed_clean: [B, 16] - Teacher embeddings on clean faces
            student_embed_occluded: [B, 16] - Student embeddings on occluded faces
            student_embed_clean: [B, 16] - Student embeddings on clean faces
            attention_map_occluded: [B, 5, 5] - Student attention on occluded faces
            attention_map_clean: [B, 5, 5] - Student attention on clean faces
            binary_mask_down: [B, 5, 5] - Binary mask downsampled (1=occluded, 0=visible)
        
        Returns:
            total_loss: Scalar
            loss_dict: Dictionary with individual loss components
        """
        # Component 1: Distillation loss (occluded pairs)
        loss_distill = self.cosine_loss(teacher_embed_clean, student_embed_occluded)
        
        # Component 2: Consistency loss (clean pairs)
        loss_consistency = self.cosine_loss(teacher_embed_clean, student_embed_clean)
        
        # Component 3: Attention regularization (occluded pairs only)
        if self.lambda3 > 0:
            loss_attention_reg = self.attention_reg(attention_map_occluded, binary_mask_down)
        else:
            loss_attention_reg = torch.tensor(0.0, device=teacher_embed_clean.device)
        
        # Component 4: Attention diversity (all pairs)
        # Combine both occluded and clean attention maps
        attention_combined = torch.cat([attention_map_occluded, attention_map_clean], dim=0)
        loss_attention_div = self.attention_div(attention_combined)
        
        # Total loss
        total_loss = (self.lambda1 * loss_distill + 
                     self.lambda2 * loss_consistency + 
                     self.lambda3 * loss_attention_reg - 
                     self.lambda4 * loss_attention_div)
        
        # Return loss dict for logging
        loss_dict = {
            'total': total_loss.item(),
            'distillation': loss_distill.item(),
            'consistency': loss_consistency.item(),
            'attention_reg': loss_attention_reg.item() if isinstance(loss_attention_reg, torch.Tensor) else 0.0,
            'attention_div': loss_attention_div.item()
        }
        
        return total_loss, loss_dict
    
    def update_lambda3(self, new_lambda3):
        """Update lambda3 for progressive curriculum"""
        self.lambda3 = new_lambda3


def compute_attention_metrics(attention_map, binary_mask_down):
    """
    Compute attention quality metrics for monitoring
    Reference: OccFECNet.md Section 6.2
    
    CRITICAL: Monitors Pearson correlation between attention and mask
    - Target correlation: 0.6-0.9 (per OccFECNet.md)
    - Too low (<0.6): Attention not learning occlusion patterns, increase lambda3
    - Too high (>0.95): Attention copying mask exactly, decrease lambda3
    
    Args:
        attention_map: [B, 5, 5] - Predicted attention
        binary_mask_down: [B, 5, 5] - Binary mask (1=occluded)
    
    Returns:
        dict with metrics:
        - correlation: Pearson correlation (TARGET: 0.6-0.9)
        - sparsity: Mean attention value
        - entropy: Spatial diversity measure
    """
    # Flatten for correlation
    attn_flat = attention_map.flatten()
    mask_flat = binary_mask_down.flatten()
    
    # Pearson correlation
    attn_mean = attn_flat.mean()
    mask_mean = mask_flat.mean()
    attn_centered = attn_flat - attn_mean
    mask_centered = mask_flat - mask_mean
    
    numerator = (attn_centered * mask_centered).sum()
    denominator = torch.sqrt((attn_centered ** 2).sum() * (mask_centered ** 2).sum())
    correlation = numerator / (denominator + 1e-8)
    
    # Sparsity (mean attention value)
    sparsity = attention_map.mean()
    
    # Entropy (spatial diversity)
    eps = 1e-8
    m = torch.clamp(attention_map, eps, 1 - eps)
    entropy = -(m * torch.log(m) + (1 - m) * torch.log(1 - m)).mean()
    
    return {
        'correlation': correlation.item(),
        'sparsity': sparsity.item(),
        'entropy': entropy.item()
    }

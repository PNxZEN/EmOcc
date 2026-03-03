"""
Debug script to check for NaN causes in Phase 4
"""
import torch
import sys

def test_gradient_explosion():
    """Test if beta parameter can explode with large gradients"""
    print("="*70)
    print("TEST 1: Beta Gradient Explosion Simulation")
    print("="*70)
    
    # Simulate Phase 3 end: beta ~ 0.003, high correlation
    beta = torch.nn.Parameter(torch.tensor(0.003))
    
    # Simulate feature and attention maps
    B, C, H, W = 4, 1792, 5, 5
    features = torch.randn(B, C, H, W, requires_grad=True)
    attention_map = torch.sigmoid(torch.randn(B, 1, H, W, requires_grad=True))
    
    # Binary mask (highly correlated with attention)
    # Correlation 0.97 means attention ≈ binary_mask
    binary_mask = (attention_map > 0.5).float()  # Perfect correlation
    binary_mask = binary_mask + 0.03 * torch.randn_like(binary_mask)  # Add small noise
    binary_mask = torch.clamp(binary_mask, 0, 1)
    
    # Residual attention formula
    attended = features + beta * features * (1 - attention_map)
    
    # Simulate loss (cosine embedding)
    target = torch.randn(B, C, H, W)
    loss = torch.nn.functional.mse_loss(attended, target)
    
    # Backward
    loss.backward()
    
    print(f"Beta value: {beta.item():.6f}")
    print(f"Beta gradient: {beta.grad.item():.6f}")
    print(f"Features gradient norm: {features.grad.norm().item():.6f}")
    
    # Simulate optimizer step with Phase 4 lr
    lr_beta = 5e-5
    with torch.no_grad():
        beta_update = lr_beta * beta.grad
        beta_new = beta - beta_update
        print(f"Beta update: {beta_update.item():.6e}")
        print(f"Beta after 1 step: {beta_new.item():.6f}")
    
    print()

def test_attention_diversity_nan():
    """Test if attention diversity loss can produce NaN"""
    print("="*70)
    print("TEST 2: Attention Diversity Entropy NaN")
    print("="*70)
    
    # Test edge cases
    test_cases = [
        ("Normal", torch.sigmoid(torch.randn(4, 5, 5))),
        ("All zeros", torch.zeros(4, 5, 5)),
        ("All ones", torch.ones(4, 5, 5)),
        ("Very small", torch.full((4, 5, 5), 1e-10)),
        ("Very large", torch.full((4, 5, 5), 1 - 1e-10)),
    ]
    
    eps = 1e-8
    for name, attention in test_cases:
        m = torch.clamp(attention, eps, 1 - eps)
        entropy = -(m * torch.log(m) + (1 - m) * torch.log(1 - m))
        entropy_mean = entropy.mean()
        
        print(f"{name:15s}: mean={entropy_mean.item():.6f}, has_nan={torch.isnan(entropy_mean).item()}")
    
    print()

def test_cosine_embedding_nan():
    """Test if cosine embedding can produce NaN"""
    print("="*70)
    print("TEST 3: Cosine Embedding Loss NaN")
    print("="*70)
    
    # Test with normal embeddings
    embed1 = torch.randn(4, 16)
    embed2 = torch.randn(4, 16)
    
    # Normalize
    embed1_norm = torch.nn.functional.normalize(embed1, p=2, dim=1)
    embed2_norm = torch.nn.functional.normalize(embed2, p=2, dim=1)
    
    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(embed1_norm, embed2_norm, dim=1)
    target = torch.ones(4)
    loss = torch.nn.functional.cosine_embedding_loss(embed1_norm, embed2_norm, target)
    
    print(f"Cosine similarity range: [{cos_sim.min().item():.4f}, {cos_sim.max().item():.4f}]")
    print(f"Loss: {loss.item():.6f}, has_nan: {torch.isnan(loss).item()}")
    
    # Test with zero embeddings (edge case)
    embed_zero = torch.zeros(4, 16)
    embed_zero[0, 0] = 1e-10  # Prevent complete zero
    embed_zero_norm = torch.nn.functional.normalize(embed_zero, p=2, dim=1)
    
    print(f"\nZero embedding norm: {embed_zero_norm.norm(dim=1)}")
    print(f"Has NaN: {torch.isnan(embed_zero_norm).any().item()}")
    
    print()

def test_high_correlation_gradients():
    """Test gradient magnitudes when attention correlation is very high"""
    print("="*70)
    print("TEST 4: High Correlation Gradient Magnitude")
    print("="*70)
    
    correlations = [0.5, 0.7, 0.9, 0.95, 0.97, 0.99]
    
    for corr in correlations:
        beta = torch.nn.Parameter(torch.tensor(0.003))
        
        # Create attention and mask with specific correlation
        binary_mask = torch.randint(0, 2, (4, 5, 5)).float()
        
        # Create attention that correlates with mask
        noise = torch.randn(4, 5, 5) * (1 - corr)
        attention = corr * binary_mask + noise
        attention = torch.clamp(torch.sigmoid(attention), 1e-8, 1 - 1e-8)
        attention.requires_grad = True
        
        # Compute actual correlation
        actual_corr = torch.corrcoef(torch.stack([
            attention.flatten(),
            binary_mask.flatten()
        ]))[0, 1].item()
        
        # Residual attention
        features = torch.randn(4, 1792, 5, 5, requires_grad=True)
        attended = features + beta * features * (1 - attention.unsqueeze(1))
        
        # Loss
        target = torch.randn_like(attended)
        loss = (attended - target).pow(2).mean()
        loss.backward()
        
        print(f"Correlation {corr:.2f} (actual: {actual_corr:.4f}): "
              f"beta_grad={beta.grad.item():.6e}, "
              f"attn_grad_norm={attention.grad.norm().item():.6e}")
    
    print()

if __name__ == "__main__":
    test_gradient_explosion()
    test_attention_diversity_nan()
    test_cosine_embedding_nan()
    test_high_correlation_gradients()

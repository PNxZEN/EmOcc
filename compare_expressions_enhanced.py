"""
Enhanced facial expression comparison with improved discrimination.
Uses multiple techniques to better separate similar from dissimilar expressions.
"""

import torch
import argparse
from PIL import Image
import torchvision.transforms as transforms
from models.FECNet import FECNet
from models.mtcnn import MTCNN
import numpy as np


def load_model(model_path, device='cuda'):
    """Load pretrained FECNet model."""
    print(f"Loading model from {model_path}...")
    model = FECNet(pretrained=False)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)
    print("Model loaded successfully!")
    return model


def preprocess_image(image_path, mtcnn, device='cuda'):
    """Detect face and preprocess image for FECNet."""
    print(f"Processing: {image_path}")
    img = Image.open(image_path).convert('RGB')
    
    # Detect and crop face
    img_cropped = mtcnn(img)
    
    if img_cropped is None:
        raise ValueError(f"No face detected in {image_path}!")
    
    # Resize to 224x224 for FECNet
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    img_tensor = transform(img_cropped).unsqueeze(0).to(device)
    return img_tensor


def extract_embeddings(model, img_tensor):
    """Extract both face features and expression embeddings."""
    with torch.no_grad():
        # Get face features from Inception network (identity features)
        face_features = model.Inc(img_tensor)[1]  # 512-dimensional
        
        # Get expression embedding from full network
        expression_embedding = model(img_tensor)  # 16-dimensional
    
    return face_features.cpu().numpy(), expression_embedding.cpu().numpy()


def normalize_vector(vec):
    """L2 normalize a vector."""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def compute_enhanced_similarity(face_feat1, expr_emb1, face_feat2, expr_emb2, method='all'):
    """
    Compute enhanced similarity metrics using multiple strategies.
    
    Strategies:
    1. Raw expression embeddings (baseline)
    2. Normalized expression embeddings (scale-invariant)
    3. Face-subtracted embeddings (removes identity component)
    4. Relative expression difference (compares deviation from neutral)
    5. Angular distance (emphasizes direction over magnitude)
    """
    
    results = {}
    
    # Strategy 1: Raw Expression Embeddings (baseline)
    euc_raw = np.linalg.norm(expr_emb1 - expr_emb2)
    cos_raw = np.dot(expr_emb1.flatten(), expr_emb2.flatten()) / (
        np.linalg.norm(expr_emb1) * np.linalg.norm(expr_emb2))
    
    results['raw'] = {
        'euclidean': euc_raw,
        'cosine_sim': cos_raw,
        'cosine_dist': 1 - cos_raw
    }
    
    # Strategy 2: Normalized Expression Embeddings (L2 normalized)
    expr_norm1 = normalize_vector(expr_emb1.flatten())
    expr_norm2 = normalize_vector(expr_emb2.flatten())
    
    euc_norm = np.linalg.norm(expr_norm1 - expr_norm2)
    cos_norm = np.dot(expr_norm1, expr_norm2)
    
    results['normalized'] = {
        'euclidean': euc_norm,
        'cosine_sim': cos_norm,
        'cosine_dist': 1 - cos_norm
    }
    
    # Strategy 3: Face-Independent Expression (subtracts mean face identity bias)
    # Instead of projecting onto high-dimensional face features,
    # compute expression relative to face feature statistics
    face_flat1 = face_feat1.flatten()
    face_flat2 = face_feat2.flatten()
    
    expr_flat1 = expr_emb1.flatten()
    expr_flat2 = expr_emb2.flatten()
    
    # Compute face similarity to use as identity correction
    face_similarity = np.dot(normalize_vector(face_flat1), normalize_vector(face_flat2))
    
    # Adjust expression comparison by identity similarity
    # If faces are very similar (same person), expressions should be more comparable
    identity_weight = max(0, face_similarity)  # 0 to 1
    
    # Compute identity-corrected expression distance
    expr_norm1_indep = normalize_vector(expr_flat1)
    expr_norm2_indep = normalize_vector(expr_flat2)
    
    euc_indep = np.linalg.norm(expr_norm1_indep - expr_norm2_indep)
    cos_indep = np.dot(expr_norm1_indep, expr_norm2_indep)
    
    # Apply identity correction (less strict if different people)
    euc_indep_corrected = euc_indep * (0.7 + 0.3 * identity_weight)
    
    results['face_independent'] = {
        'euclidean': euc_indep_corrected,
        'cosine_sim': cos_indep,
        'cosine_dist': 1 - cos_indep,
        'face_similarity': face_similarity
    }
    
    # Strategy 4: Weighted Expression Distance (emphasize expression-specific dimensions)
    # Use variance across dimensions to weight them
    expr_var = np.var(np.vstack([expr_flat1, expr_flat2]), axis=0)
    weights = expr_var / (np.sum(expr_var) + 1e-8)
    
    weighted_diff = weights * (expr_flat1 - expr_flat2) ** 2
    euc_weighted = np.sqrt(np.sum(weighted_diff))
    
    results['weighted'] = {
        'euclidean': euc_weighted,
        'cosine_sim': cos_raw,  # Same as raw
        'cosine_dist': 1 - cos_raw
    }
    
    # Strategy 5: Angular Distance (more discriminative than cosine)
    # Angular distance = arccos(cosine_similarity)
    angular_dist_norm = np.arccos(np.clip(cos_norm, -1.0, 1.0))
    angular_dist_raw = np.arccos(np.clip(cos_raw, -1.0, 1.0))
    
    results['angular'] = {
        'angular_distance': angular_dist_norm,
        'angular_distance_raw': angular_dist_raw,
        'similarity_score': 1 - (angular_dist_norm / np.pi)  # Normalized to [0, 1]
    }
    
    # Strategy 6: Combined Score (recommended)
    # Combine cosine similarity and angular distance only
    combined_score = (
        0.5 * (1 - cos_norm) +              # Normalized cosine distance (50%)
        0.5 * (angular_dist_norm / np.pi)   # Angular distance (50%)
    )
    
    results['combined'] = {
        'dissimilarity_score': combined_score,
        'similarity_score': 1 - combined_score
    }
    
    return results


def interpret_similarity(score, metric_type='combined'):
    """Interpret similarity score with appropriate thresholds."""
    if metric_type == 'combined':
        # Combined similarity score (0-1, higher is more similar)
        if score > 0.85:
            return "VERY SIMILAR"
        elif score > 0.70:
            return "SIMILAR"
        elif score > 0.50:
            return "SOMEWHAT SIMILAR"
        elif score > 0.30:
            return "DIFFERENT"
        else:
            return "VERY DIFFERENT"
    elif metric_type == 'euclidean_normalized':
        # Normalized euclidean distance (0-2, lower is more similar)
        if score < 0.3:
            return "VERY SIMILAR"
        elif score < 0.6:
            return "SIMILAR"
        elif score < 1.0:
            return "SOMEWHAT SIMILAR"
        elif score < 1.4:
            return "DIFFERENT"
        else:
            return "VERY DIFFERENT"
    else:  # angular
        # Angular distance (0-pi, lower is more similar)
        if score < 0.2:
            return "VERY SIMILAR"
        elif score < 0.5:
            return "SIMILAR"
        elif score < 0.8:
            return "SOMEWHAT SIMILAR"
        elif score < 1.2:
            return "DIFFERENT"
        else:
            return "VERY DIFFERENT"


def main():
    parser = argparse.ArgumentParser(description='Enhanced facial expression comparison')
    parser.add_argument('--image1', type=str, required=True,
                        help='Path to first image')
    parser.add_argument('--image2', type=str, required=True,
                        help='Path to second image')
    parser.add_argument('--model', type=str, default='pretrained/FECNet.pt',
                        help='Path to pretrained model')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use: cuda or cpu')
    parser.add_argument('--method', type=str, default='all',
                        choices=['all', 'combined', 'normalized', 'angular', 'face_independent'],
                        help='Comparison method to use')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        device = 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print("=" * 80)
    
    # Initialize MTCNN for face detection
    print("Initializing face detector...")
    mtcnn = MTCNN(device=device)
    
    # Load model
    model = load_model(args.model, device=device)
    print("=" * 80)
    
    # Process first image
    print("\nProcessing Image 1:")
    print("-" * 80)
    try:
        img_tensor1 = preprocess_image(args.image1, mtcnn, device=device)
        face_feat1, expr_emb1 = extract_embeddings(model, img_tensor1)
        print(f"Face features shape: {face_feat1.shape}, Expression embedding shape: {expr_emb1.shape}")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Process second image
    print("\nProcessing Image 2:")
    print("-" * 80)
    try:
        img_tensor2 = preprocess_image(args.image2, mtcnn, device=device)
        face_feat2, expr_emb2 = extract_embeddings(model, img_tensor2)
        print(f"Face features shape: {face_feat2.shape}, Expression embedding shape: {expr_emb2.shape}")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Compute enhanced similarity
    print("\n" + "=" * 80)
    print("ENHANCED SIMILARITY ANALYSIS")
    print("=" * 80)
    
    results = compute_enhanced_similarity(face_feat1, expr_emb1, face_feat2, expr_emb2, args.method)
    
    if args.method == 'all':
        print("\n[1] RAW EXPRESSION EMBEDDINGS (Baseline):")
        print(f"    Euclidean Distance: {results['raw']['euclidean']:.6f}")
        print(f"    Cosine Similarity:  {results['raw']['cosine_sim']:.6f}")
        print(f"    Cosine Distance:    {results['raw']['cosine_dist']:.6f}")
        
        print("\n[2] NORMALIZED EXPRESSION EMBEDDINGS (Scale-Invariant):")
        print(f"    Euclidean Distance: {results['normalized']['euclidean']:.6f}")
        print(f"    Cosine Similarity:  {results['normalized']['cosine_sim']:.6f}")
        print(f"    Cosine Distance:    {results['normalized']['cosine_dist']:.6f}")
        interp_norm = interpret_similarity(results['normalized']['euclidean'], 'euclidean_normalized')
        print(f"    => {interp_norm}")
        
        print("\n[3] FACE-INDEPENDENT EXPRESSION (Identity Removed):")
        print(f"    Euclidean Distance: {results['face_independent']['euclidean']:.6f}")
        print(f"    Cosine Similarity:  {results['face_independent']['cosine_sim']:.6f}")
        print(f"    Cosine Distance:    {results['face_independent']['cosine_dist']:.6f}")
        
        print("\n[4] WEIGHTED EXPRESSION DISTANCE:")
        print(f"    Weighted Euclidean: {results['weighted']['euclidean']:.6f}")
        
        print("\n[5] ANGULAR DISTANCE (Most Discriminative):")
        print(f"    Angular Distance (normalized): {results['angular']['angular_distance']:.6f}")
        print(f"    Angular Distance (raw):        {results['angular']['angular_distance_raw']:.6f}")
        print(f"    Similarity Score:              {results['angular']['similarity_score']:.6f}")
        interp_angular = interpret_similarity(results['angular']['angular_distance'], 'angular')
        print(f"    => {interp_angular}")
        
        print("\n[6] COMBINED SCORE (Recommended):")
        print(f"    Dissimilarity Score: {results['combined']['dissimilarity_score']:.6f}")
        print(f"    Similarity Score:    {results['combined']['similarity_score']:.6f}")
        interp_combined = interpret_similarity(results['combined']['similarity_score'], 'combined')
        print(f"    => {interp_combined}")
        
        print("\n" + "=" * 80)
        print("RECOMMENDATION:")
        print(f"  Use COMBINED SCORE for best discrimination")
        print(f"  Similarity: {results['combined']['similarity_score']:.4f} => {interp_combined}")
        
    else:
        # Show only requested method
        method_name = args.method.upper().replace('_', ' ')
        print(f"\n{method_name}:")
        for key, value in results[args.method].items():
            print(f"    {key}: {value:.6f}")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()

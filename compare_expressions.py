"""
Compare facial expressions between two images using FECNet embeddings.
This script extracts expression embeddings from two face images and computes their similarity.
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


def extract_embedding(model, img_tensor):
    """Extract facial expression embedding."""
    with torch.no_grad():
        embedding = model(img_tensor)
    return embedding.cpu().numpy()


def compute_similarity(embedding1, embedding2):
    """Compute similarity metrics between two embeddings."""
    # Euclidean distance (L2 distance)
    euclidean_dist = np.linalg.norm(embedding1 - embedding2)
    
    # Cosine similarity
    dot_product = np.dot(embedding1.flatten(), embedding2.flatten())
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    cosine_sim = dot_product / (norm1 * norm2)
    
    # Cosine distance (1 - cosine similarity)
    cosine_dist = 1 - cosine_sim
    
    return {
        'euclidean_distance': euclidean_dist,
        'cosine_similarity': cosine_sim,
        'cosine_distance': cosine_dist
    }


def main():
    parser = argparse.ArgumentParser(description='Compare facial expressions between two images')
    parser.add_argument('--image1', type=str, required=True,
                        help='Path to first image')
    parser.add_argument('--image2', type=str, required=True,
                        help='Path to second image')
    parser.add_argument('--model', type=str, default='pretrained/FECNet.pt',
                        help='Path to pretrained model')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use: cuda or cpu')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        device = 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print("=" * 70)
    
    # Initialize MTCNN for face detection
    print("Initializing face detector...")
    mtcnn = MTCNN(device=device)
    
    # Load model
    model = load_model(args.model, device=device)
    print("=" * 70)
    
    # Process first image
    print("\nProcessing Image 1:")
    print("-" * 70)
    try:
        img_tensor1 = preprocess_image(args.image1, mtcnn, device=device)
        embedding1 = extract_embedding(model, img_tensor1)
        print(f"✓ Face detected and embedding extracted")
        print(f"  Embedding: {embedding1.flatten()[:4]}... (showing first 4 of 16 dims)")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # Process second image
    print("\nProcessing Image 2:")
    print("-" * 70)
    try:
        img_tensor2 = preprocess_image(args.image2, mtcnn, device=device)
        embedding2 = extract_embedding(model, img_tensor2)
        print(f"✓ Face detected and embedding extracted")
        print(f"  Embedding: {embedding2.flatten()[:4]}... (showing first 4 of 16 dims)")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # Compute similarity
    print("\n" + "=" * 70)
    print("SIMILARITY ANALYSIS")
    print("=" * 70)
    
    similarity = compute_similarity(embedding1, embedding2)
    
    print(f"\n📊 Similarity Metrics:")
    print(f"  • Euclidean Distance:  {similarity['euclidean_distance']:.6f}")
    print(f"  • Cosine Similarity:   {similarity['cosine_similarity']:.6f}")
    print(f"  • Cosine Distance:     {similarity['cosine_distance']:.6f}")
    
    # Interpretation
    print(f"\n💡 Interpretation:")
    
    # Euclidean distance interpretation
    euc_dist = similarity['euclidean_distance']
    if euc_dist < 0.3:
        euc_interp = "Very Similar"
        euc_emoji = "😀"
    elif euc_dist < 0.6:
        euc_interp = "Similar"
        euc_emoji = "🙂"
    elif euc_dist < 1.0:
        euc_interp = "Somewhat Similar"
        euc_emoji = "😐"
    elif euc_dist < 1.5:
        euc_interp = "Different"
        euc_emoji = "😕"
    else:
        euc_interp = "Very Different"
        euc_emoji = "😟"
    
    print(f"  Euclidean Distance: {euc_interp} {euc_emoji}")
    print(f"    (Lower is more similar, typical range: 0.0-2.0)")
    
    # Cosine similarity interpretation
    cos_sim = similarity['cosine_similarity']
    if cos_sim > 0.95:
        cos_interp = "Very Similar"
        cos_emoji = "😀"
    elif cos_sim > 0.85:
        cos_interp = "Similar"
        cos_emoji = "🙂"
    elif cos_sim > 0.70:
        cos_interp = "Somewhat Similar"
        cos_emoji = "😐"
    elif cos_sim > 0.50:
        cos_interp = "Different"
        cos_emoji = "😕"
    else:
        cos_interp = "Very Different"
        cos_emoji = "😟"
    
    print(f"  Cosine Similarity: {cos_interp} {cos_emoji}")
    print(f"    (Higher is more similar, range: -1.0 to 1.0)")
    
    # Overall assessment
    print(f"\n📋 Overall Assessment:")
    if euc_dist < 0.5 and cos_sim > 0.85:
        print(f"  The facial expressions are VERY SIMILAR! 🎯")
    elif euc_dist < 1.0 and cos_sim > 0.70:
        print(f"  The facial expressions are SIMILAR. ✓")
    elif euc_dist < 1.5:
        print(f"  The facial expressions are SOMEWHAT DIFFERENT. ~")
    else:
        print(f"  The facial expressions are QUITE DIFFERENT. ✗")
    
    print("\n" + "=" * 70)
    print("Analysis complete!")


if __name__ == '__main__':
    main()

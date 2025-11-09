"""
FECNet Inference Script
Run inference on a single image to extract facial expression embeddings.
"""

import torch
import argparse
from PIL import Image
import torchvision.transforms as transforms
from models.FECNet import FECNet
from models.mtcnn import MTCNN
import numpy as np


def load_model(model_path, device='cuda'):
    """Load pretrained FECNet model.
    
    Args:
        model_path: Path to the pretrained model weights
        device: 'cuda' or 'cpu'
    
    Returns:
        Loaded model in eval mode
    """
    print(f"Loading model from {model_path}...")
    model = FECNet(pretrained=False)
    
    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.eval()
    model = model.to(device)
    print("Model loaded successfully!")
    return model


def preprocess_image(image_path, mtcnn, device='cuda'):
    """Detect face and preprocess image for FECNet.
    
    Args:
        image_path: Path to input image
        mtcnn: MTCNN face detector
        device: 'cuda' or 'cpu'
    
    Returns:
        Preprocessed image tensor ready for model input
    """
    print(f"Loading image from {image_path}...")
    img = Image.open(image_path).convert('RGB')
    
    # Detect and crop face
    print("Detecting face...")
    img_cropped = mtcnn(img)
    
    if img_cropped is None:
        raise ValueError("No face detected in the image!")
    
    print(f"Face detected! Cropped size: {img_cropped.shape}")
    
    # Resize to 224x224 for FECNet
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    img_tensor = transform(img_cropped).unsqueeze(0).to(device)
    print(f"Image preprocessed to shape: {img_tensor.shape}")
    
    return img_tensor


def extract_embedding(model, img_tensor):
    """Extract facial expression embedding.
    
    Args:
        model: FECNet model
        img_tensor: Preprocessed image tensor
    
    Returns:
        16-dimensional embedding vector
    """
    print("Extracting expression embedding...")
    with torch.no_grad():
        embedding = model(img_tensor)
    
    return embedding.cpu().numpy()


def compute_similarity(embedding1, embedding2):
    """Compute similarity between two embeddings using Euclidean distance.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
    
    Returns:
        Euclidean distance (lower means more similar)
    """
    return np.linalg.norm(embedding1 - embedding2)


def main():
    parser = argparse.ArgumentParser(description='FECNet Inference')
    parser.add_argument('--image', type=str, default='examples/nikhil_face.jpg',
                        help='Path to input image')
    parser.add_argument('--model', type=str, default='pretrained/FECNet.pt',
                        help='Path to pretrained model')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use: cuda or cpu')
    parser.add_argument('--compare', type=str, default=None,
                        help='Optional: Path to second image for expression comparison')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        device = 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print("-" * 60)
    
    # Initialize MTCNN for face detection
    print("Initializing face detector...")
    mtcnn = MTCNN(device=device)
    print("-" * 60)
    
    # Load model
    model = load_model(args.model, device=device)
    print("-" * 60)
    
    # Process first image
    img_tensor = preprocess_image(args.image, mtcnn, device=device)
    embedding = extract_embedding(model, img_tensor)
    
    print("-" * 60)
    print(f"Expression Embedding (16-dim):")
    print(embedding)
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
    
    # If comparison image provided
    if args.compare:
        print("-" * 60)
        print(f"Processing comparison image: {args.compare}")
        img_tensor2 = preprocess_image(args.compare, mtcnn, device=device)
        embedding2 = extract_embedding(model, img_tensor2)
        
        print("-" * 60)
        print(f"Comparison Expression Embedding (16-dim):")
        print(embedding2)
        
        # Compute similarity
        distance = compute_similarity(embedding, embedding2)
        print("-" * 60)
        print(f"Expression Similarity:")
        print(f"  Euclidean Distance: {distance:.4f}")
        print(f"  (Lower distance = more similar expressions)")
    
    print("-" * 60)
    print("Inference complete!")


if __name__ == '__main__':
    main()

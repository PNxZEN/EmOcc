"""
Real-time facial expression comparison with webcam.
Compares expressions in webcam frames with a target image.
"""

import torch
import argparse
import cv2
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from models.FECNet import FECNet
from models.mtcnn import MTCNN
import numpy as np
import time


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


def preprocess_frame(frame, mtcnn, device='cuda'):
    """Detect face and preprocess frame for FECNet."""
    # Convert BGR (OpenCV) to RGB (PIL)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    
    # Detect and crop face
    img_cropped = mtcnn(pil_img)
    
    if img_cropped is None:
        return None
    
    # Resize to 224x224 for FECNet
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    img_tensor = transform(img_cropped).unsqueeze(0).to(device)
    return img_tensor


def preprocess_image(image_path, mtcnn: MTCNN, device='cuda'):
    """Load and preprocess target image."""
    print(f"Loading target image: {image_path}")
    img = Image.open(image_path).convert('RGB')
    
    # Detect and crop face
    img_cropped : MTCNN = mtcnn(img)
    
    if img_cropped is None:
        raise ValueError(f"No face detected in {image_path}!")
    
    print("Target face detected successfully!")

    # Show target face
    img_cropped_pil = transforms.ToPILImage()(img_cropped)
    img_cropped_cv2 = cv2.cvtColor(np.array(img_cropped_pil), cv2.COLOR_RGB2BGR)
    cv2.imshow('Target Face (Press any key to continue)', img_cropped_cv2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
    # Euclidean distance
    euclidean_dist = np.linalg.norm(embedding1 - embedding2)
    
    # Cosine similarity
    dot_product = np.dot(embedding1.flatten(), embedding2.flatten())
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    cosine_sim = dot_product / (norm1 * norm2)
    
    return euclidean_dist, cosine_sim


def get_similarity_text(euc_dist, cos_sim):
    """Get similarity assessment text."""
    if euc_dist < 0.5 and cos_sim > 0.85:
        return "VERY SIMILAR"
    elif euc_dist < 1.0 and cos_sim > 0.70:
        return "SIMILAR"
    elif euc_dist < 1.5:
        return "SOMEWHAT DIFFERENT"
    else:
        return "DIFFERENT"


def main():
    parser = argparse.ArgumentParser(description='Real-time expression comparison with webcam')
    parser.add_argument('--target', type=str, required=True,
                        help='Path to target image for comparison')
    parser.add_argument('--model', type=str, default='pretrained/FECNet.pt',
                        help='Path to pretrained model')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use: cuda or cpu')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device index (default: 0)')
    parser.add_argument('--frame-skip', type=int, default=3,
                        help='Process every Nth frame (default: 3)')
    parser.add_argument('--width', type=int, default=640,
                        help='Webcam frame width (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                        help='Webcam frame height (default: 480)')
    
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
    
    # Load and process target image
    print("\nProcessing target image...")
    print("-" * 70)
    try:
        target_tensor = preprocess_image(args.target, mtcnn, device=device)
        target_embedding = extract_embedding(model, target_tensor)
        print("Target embedding extracted successfully!")
    except Exception as e:
        print(f"Error processing target image: {e}")
        return
    
    print("\n" + "=" * 70)
    print("Starting webcam stream...")
    print("Press 'q' to quit")
    print("=" * 70 + "\n")
    
    # Initialize webcam
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        return
    
    frame_count = 0
    last_similarity = None
    last_euc_dist = None
    last_cos_sim = None
    fps_time = time.time()
    fps_counter = 0
    fps = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame")
                break
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter >= 30:
                fps = fps_counter / (time.time() - fps_time)
                fps_time = time.time()
                fps_counter = 0
            
            # Process every Nth frame
            if frame_count % args.frame_skip == 0:
                try:
                    # Preprocess frame
                    frame_tensor = preprocess_frame(frame, mtcnn, device=device)
                    
                    if frame_tensor is not None:
                        # Extract embedding
                        frame_embedding = extract_embedding(model, frame_tensor)
                        
                        # Compute similarity
                        euc_dist, cos_sim = compute_similarity(target_embedding, frame_embedding)
                        
                        # Get similarity text
                        similarity_text = get_similarity_text(euc_dist, cos_sim)
                        
                        # Store for display
                        last_similarity = similarity_text
                        last_euc_dist = euc_dist
                        last_cos_sim = cos_sim
                    else:
                        last_similarity = "NO FACE DETECTED"
                        last_euc_dist = None
                        last_cos_sim = None
                        
                except Exception as e:
                    last_similarity = f"ERROR: {str(e)[:30]}"
                    last_euc_dist = None
                    last_cos_sim = None
            
            # Display information on frame
            display_frame = frame.copy()
            
            # Add semi-transparent overlay at top
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
            
            # Display FPS
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame info
            cv2.putText(display_frame, f"Frame: {frame_count} (Processing every {args.frame_skip})", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display similarity results
            if last_similarity:
                color = (0, 255, 0) if "SIMILAR" in last_similarity else (0, 165, 255)
                if "NO FACE" in last_similarity or "ERROR" in last_similarity:
                    color = (0, 0, 255)
                
                cv2.putText(display_frame, f"Similarity: {last_similarity}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                if last_euc_dist is not None:
                    metrics_text = f"Euclidean: {last_euc_dist:.3f} | Cosine: {last_cos_sim:.3f}"
                    cv2.putText(display_frame, metrics_text, 
                               (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow('Expression Comparison - Press q to quit', display_frame)
            
            frame_count += 1
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("\nWebcam stream ended")
        print(f"Total frames processed: {frame_count}")
        print(f"Frames analyzed: {frame_count // args.frame_skip}")


if __name__ == '__main__':
    main()

"""
Calculate Face Visibility Percentage with Facial Feature Analysis

This script compares face regions (detected using MTCNN) with occlusion masks
to determine what percentage of each face remains visible after occlusion.
It also analyzes if critical facial features (eyes, mouth) are blocked.

Output: CSV file with image paths, visibility percentages, and feature occlusion status
"""

import os
import cv2
import numpy as np
import pandas as pd
from mtcnn import MTCNN
from tqdm import tqdm
import mediapipe as mp

# Configuration
PROJECT_PATH = "./output"
ORIGINAL_BASE = os.path.join(PROJECT_PATH, "RAF-DB/train")
OCCLUDED_BASE = os.path.join(PROJECT_PATH, "RAF-DB_Occluded/train")
OUTPUT_CSV = os.path.join(PROJECT_PATH, "face_visibility_with_features.csv")

# Emotion categories (1-7)
EMOTIONS = list(range(1, 8))

# Initialize MTCNN face detector and MediaPipe Face Mesh
print("Initializing MTCNN face detector and MediaPipe Face Mesh...")
detector = MTCNN()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# MediaPipe landmark indices for facial features
LEFT_EYE_INDICES = [33, 133, 160, 158, 144, 153]
RIGHT_EYE_INDICES = [362, 263, 387, 386, 374, 380]
MOUTH_INDICES = [61, 91, 181, 84, 17, 314, 405, 321, 375, 291]
print("Initialization complete.")

def get_face_bounding_box(image_path):
    """
    Detect face in image and return bounding box coordinates.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Bounding box as (x, y, width, height) or None if no face detected
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Convert BGR to RGB for MTCNN
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        detections = detector.detect_faces(img_rgb)
        
        if len(detections) == 0:
            return None
        
        # Get the first (most confident) detection
        bbox = detections[0]['box']
        return bbox  # Returns (x, y, width, height)
    except Exception as e:
        return None

def get_facial_landmarks(image_path):
    """
    Detect facial landmarks using MediaPipe Face Mesh.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with landmark coordinates for eyes and mouth, or None if detection fails
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Check if image has valid dimensions
        h, w = img.shape[:2]
        if h < 48 or w < 48:  # MediaPipe requires minimum size
            return None
        
        # Convert BGR to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect facial landmarks
        results = face_mesh.process(img_rgb)
        
        if not results.multi_face_landmarks:
            return None
        
        landmarks = results.multi_face_landmarks[0].landmark
    except Exception as e:
        # Silently catch MediaPipe/TensorFlow errors
        return None
    
    # Extract eye and mouth landmarks
    left_eye_points = []
    right_eye_points = []
    mouth_points = []
    
    for idx in LEFT_EYE_INDICES:
        lm = landmarks[idx]
        left_eye_points.append((int(lm.x * w), int(lm.y * h)))
    
    for idx in RIGHT_EYE_INDICES:
        lm = landmarks[idx]
        right_eye_points.append((int(lm.x * w), int(lm.y * h)))
    
    for idx in MOUTH_INDICES:
        lm = landmarks[idx]
        mouth_points.append((int(lm.x * w), int(lm.y * h)))
    
    return {
        'left_eye': left_eye_points,
        'right_eye': right_eye_points,
        'mouth': mouth_points
    }

def check_feature_occlusion(feature_points, occlusion_mask):
    """
    Check if a facial feature is completely or partially occluded.
    
    Args:
        feature_points: List of (x, y) coordinates defining the feature
        occlusion_mask: Grayscale occlusion mask (white=occluded, black=visible)
        
    Returns:
        Dictionary with occlusion statistics for the feature
    """
    if not feature_points or len(feature_points) == 0:
        return None
    
    # Create a mask for the feature region
    feature_points_np = np.array(feature_points, dtype=np.int32)
    
    # Create a polygon mask for the feature
    mask = np.zeros(occlusion_mask.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, feature_points_np, 255)
    
    # Extract the feature region from occlusion mask
    feature_region = cv2.bitwise_and(occlusion_mask, mask)
    
    # Count pixels in the feature region
    total_pixels = np.sum(mask > 0)
    if total_pixels == 0:
        return None
    
    # Count occluded pixels (white in occlusion mask)
    occluded_pixels = np.sum(feature_region > 128)
    visible_pixels = total_pixels - occluded_pixels
    
    occlusion_percentage = (occluded_pixels / total_pixels) * 100
    visibility_percentage = (visible_pixels / total_pixels) * 100
    
    return {
        'total_pixels': int(total_pixels),
        'occluded_pixels': int(occluded_pixels),
        'visible_pixels': int(visible_pixels),
        'occlusion_percentage': float(occlusion_percentage),
        'visibility_percentage': float(visibility_percentage),
        'is_completely_blocked': visibility_percentage < 5,  # <5% visible = blocked
        'is_partially_visible': 5 <= visibility_percentage < 80  # 5-80% = partial
    }

def calculate_visibility_percentage(face_bbox, occlusion_mask_path, image_shape, landmarks=None):
    """
    Calculate what percentage of the face region is visible (not occluded).
    Optionally check facial features if landmarks are provided.
    
    Args:
        face_bbox: Bounding box (x, y, width, height) of the face
        occlusion_mask_path: Path to the occlusion mask image
        image_shape: Shape of the original image (height, width)
        landmarks: Optional dictionary with facial landmarks
        
    Returns:
        Dictionary with visibility metrics and feature occlusion status
    """
    if face_bbox is None:
        return None
    
    x, y, w, h = face_bbox
    
    # Ensure coordinates are within image bounds
    x = max(0, x)
    y = max(0, y)
    w = min(w, image_shape[1] - x)
    h = min(h, image_shape[0] - y)
    
    # Load occlusion mask
    occlusion_mask = cv2.imread(occlusion_mask_path, cv2.IMREAD_GRAYSCALE)
    
    if occlusion_mask is None:
        return None
    
    # Resize mask to match original image size if needed
    if occlusion_mask.shape[:2] != image_shape[:2]:
        occlusion_mask = cv2.resize(occlusion_mask, (image_shape[1], image_shape[0]), 
                                     interpolation=cv2.INTER_NEAREST)
    
    # Extract the face region from the occlusion mask
    face_region_mask = occlusion_mask[y:y+h, x:x+w]
    
    # In the occlusion mask, white (255) = occluded, black (0) = visible
    # Calculate total pixels in face region
    total_pixels = face_region_mask.size
    
    if total_pixels == 0:
        return None
    
    # Count visible pixels (black pixels, value close to 0)
    # Use threshold to handle anti-aliasing
    visible_pixels = np.sum(face_region_mask < 128)
    
    # Calculate percentage
    visibility_percentage = (visible_pixels / total_pixels) * 100
    
    result = {
        'face_visibility_percentage': visibility_percentage
    }
    
    # Check facial features if landmarks are provided
    if landmarks:
        left_eye_status = check_feature_occlusion(landmarks.get('left_eye'), occlusion_mask)
        right_eye_status = check_feature_occlusion(landmarks.get('right_eye'), occlusion_mask)
        mouth_status = check_feature_occlusion(landmarks.get('mouth'), occlusion_mask)
        
        result['left_eye'] = left_eye_status
        result['right_eye'] = right_eye_status
        result['mouth'] = mouth_status
    
    return result

def process_all_images():
    """
    Process all images across all emotion categories and generate CSV report.
    Note: Only processes images that have been occluded (exist in the occluded directory).
    """
    results = []
    
    print("\nProcessing images...")
    print("Note: Only analyzing images that have occlusion masks (not all original images may be occluded)\n")
    
    for emotion_idx in EMOTIONS:
        emotion_str = str(emotion_idx)
        
        print(f"\n{'='*50}")
        print(f"Processing Emotion {emotion_idx}")
        print(f"{'='*50}")
        
        # Paths
        original_dir = os.path.join(ORIGINAL_BASE, emotion_str)
        occluded_img_dir = os.path.join(OCCLUDED_BASE, emotion_str, "hands", "img")
        occlusion_mask_dir = os.path.join(OCCLUDED_BASE, emotion_str, "hands", "occlusion_mask")
        
        # Check if directories exist
        if not os.path.exists(original_dir):
            print(f"  Warning: Original directory not found: {original_dir}")
            continue
        
        if not os.path.exists(occluded_img_dir):
            print(f"  Warning: Occluded directory not found: {occluded_img_dir}")
            continue
        
        if not os.path.exists(occlusion_mask_dir):
            print(f"  Warning: Occlusion mask directory not found: {occlusion_mask_dir}")
            continue
        
        # Count images for comparison
        original_count = len([f for f in os.listdir(original_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        occluded_images = [f for f in os.listdir(occluded_img_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        occluded_count = len(occluded_images)
        
        print(f"Original images: {original_count}")
        print(f"Occluded images: {occluded_count}")
        if occluded_count < original_count:
            print(f"Note: {original_count - occluded_count} images were not occluded")
        
        # Process each image
        successful = 0
        failed = 0
        
        for img_name in tqdm(occluded_images, desc=f"Emotion {emotion_idx}"):
            # The original image has .png extension, but occluded might be .jpg
            base_name = os.path.splitext(img_name)[0]
            
            # Try to find the original image (could be .png or .jpg)
            original_img_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                potential_path = os.path.join(original_dir, base_name + ext)
                if os.path.exists(potential_path):
                    original_img_path = potential_path
                    break
            
            occluded_img_path = os.path.join(occluded_img_dir, img_name)
            occlusion_mask_path = os.path.join(occlusion_mask_dir, base_name + ".png")
            
            # Check if all files exist
            if original_img_path is None:
                print(f"  Warning: Original image not found for: {base_name}")
                failed += 1
                results.append({
                    'emotion': emotion_idx,
                    'image_name': img_name,
                    'original_path': 'NOT_FOUND',
                    'occluded_path': occluded_img_path,
                    'occlusion_mask_path': occlusion_mask_path,
                    'face_detected': False,
                    'face_visibility_percentage': None,
                    'left_eye_visible_pct': None,
                    'left_eye_blocked': None,
                    'right_eye_visible_pct': None,
                    'right_eye_blocked': None,
                    'mouth_visible_pct': None,
                    'mouth_blocked': None,
                    'both_eyes_blocked': None,
                    'eyes_and_mouth_blocked': None,
                    'error': 'Original image not found'
                })
                continue
            
            if not os.path.exists(occlusion_mask_path):
                print(f"  Warning: Occlusion mask not found: {occlusion_mask_path}")
                failed += 1
                results.append({
                    'emotion': emotion_idx,
                    'image_name': img_name,
                    'original_path': original_img_path,
                    'occluded_path': occluded_img_path,
                    'occlusion_mask_path': occlusion_mask_path,
                    'face_detected': False,
                    'face_visibility_percentage': None,
                    'left_eye_visible_pct': None,
                    'left_eye_blocked': None,
                    'right_eye_visible_pct': None,
                    'right_eye_blocked': None,
                    'mouth_visible_pct': None,
                    'mouth_blocked': None,
                    'both_eyes_blocked': None,
                    'eyes_and_mouth_blocked': None,
                    'error': 'Occlusion mask not found'
                })
                continue
            
            # Detect face in original image
            face_bbox = get_face_bounding_box(original_img_path)
            
            if face_bbox is None:
                failed += 1
                results.append({
                    'emotion': emotion_idx,
                    'image_name': img_name,
                    'original_path': original_img_path,
                    'occluded_path': occluded_img_path,
                    'occlusion_mask_path': occlusion_mask_path,
                    'face_detected': False,
                    'face_visibility_percentage': None,
                    'left_eye_visible_pct': None,
                    'left_eye_blocked': None,
                    'right_eye_visible_pct': None,
                    'right_eye_blocked': None,
                    'mouth_visible_pct': None,
                    'mouth_blocked': None,
                    'both_eyes_blocked': None,
                    'eyes_and_mouth_blocked': None,
                    'error': 'No face detected'
                })
                continue
            
            # Get facial landmarks
            landmarks = get_facial_landmarks(original_img_path)
            
            # Get image shape
            img = cv2.imread(original_img_path)
            image_shape = img.shape
            
            # Calculate visibility with feature analysis
            visibility_result = calculate_visibility_percentage(
                face_bbox, occlusion_mask_path, image_shape, landmarks
            )
            
            if visibility_result is None:
                failed += 1
                results.append({
                    'emotion': emotion_idx,
                    'image_name': img_name,
                    'original_path': original_img_path,
                    'occluded_path': occluded_img_path,
                    'occlusion_mask_path': occlusion_mask_path,
                    'face_detected': True,
                    'face_bbox_x': face_bbox[0],
                    'face_bbox_y': face_bbox[1],
                    'face_bbox_width': face_bbox[2],
                    'face_bbox_height': face_bbox[3],
                    'face_visibility_percentage': None,
                    'left_eye_visible_pct': None,
                    'left_eye_blocked': None,
                    'right_eye_visible_pct': None,
                    'right_eye_blocked': None,
                    'mouth_visible_pct': None,
                    'mouth_blocked': None,
                    'both_eyes_blocked': None,
                    'eyes_and_mouth_blocked': None,
                    'error': 'Failed to calculate visibility'
                })
            else:
                successful += 1
                
                # Extract feature occlusion data
                left_eye = visibility_result.get('left_eye')
                right_eye = visibility_result.get('right_eye')
                mouth = visibility_result.get('mouth')
                
                left_eye_blocked = left_eye.get('is_completely_blocked') if left_eye else None
                right_eye_blocked = right_eye.get('is_completely_blocked') if right_eye else None
                mouth_blocked = mouth.get('is_completely_blocked') if mouth else None
                
                both_eyes_blocked = (left_eye_blocked and right_eye_blocked) if (left_eye_blocked is not None and right_eye_blocked is not None) else None
                eyes_and_mouth_blocked = (left_eye_blocked and right_eye_blocked and mouth_blocked) if (left_eye_blocked is not None and right_eye_blocked is not None and mouth_blocked is not None) else None
                
                # Store results
                results.append({
                    'emotion': emotion_idx,
                    'image_name': img_name,
                    'original_path': original_img_path,
                    'occluded_path': occluded_img_path,
                    'occlusion_mask_path': occlusion_mask_path,
                    'face_detected': True,
                    'landmarks_detected': landmarks is not None,
                    'face_bbox_x': face_bbox[0],
                    'face_bbox_y': face_bbox[1],
                    'face_bbox_width': face_bbox[2],
                    'face_bbox_height': face_bbox[3],
                    'face_visibility_percentage': visibility_result.get('face_visibility_percentage'),
                    'left_eye_visible_pct': left_eye.get('visibility_percentage') if left_eye else None,
                    'left_eye_blocked': left_eye_blocked,
                    'right_eye_visible_pct': right_eye.get('visibility_percentage') if right_eye else None,
                    'right_eye_blocked': right_eye_blocked,
                    'mouth_visible_pct': mouth.get('visibility_percentage') if mouth else None,
                    'mouth_blocked': mouth_blocked,
                    'both_eyes_blocked': both_eyes_blocked,
                    'eyes_and_mouth_blocked': eyes_and_mouth_blocked,
                    'error': None
                })
        
        print(f"Emotion {emotion_idx} complete: {successful} successful, {failed} failed")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Results saved to: {OUTPUT_CSV}")
    print(f"Total images processed: {len(results)}")
    
    if len(df) > 0:
        # Print summary statistics
        detected = df[df['face_detected'] == True]
        print(f"\nFace detection success rate: {len(detected)}/{len(df)} ({len(detected)/len(df)*100:.1f}%)")
        
        if len(detected) > 0:
            # Face visibility statistics
            valid_visibility = detected[detected['face_visibility_percentage'].notna()]
            print(f"\nFace Visibility Statistics:")
            print(f"  Mean visibility: {valid_visibility['face_visibility_percentage'].mean():.2f}%")
            print(f"  Median visibility: {valid_visibility['face_visibility_percentage'].median():.2f}%")
            print(f"  Min visibility: {valid_visibility['face_visibility_percentage'].min():.2f}%")
            print(f"  Max visibility: {valid_visibility['face_visibility_percentage'].max():.2f}%")
            print(f"  Std deviation: {valid_visibility['face_visibility_percentage'].std():.2f}%")
            
            # Feature occlusion statistics
            landmarks_detected = detected[detected['landmarks_detected'] == True]
            if len(landmarks_detected) > 0:
                print(f"\nFacial Feature Occlusion Analysis:")
                print(f"  Landmarks detected: {len(landmarks_detected)}/{len(detected)} ({len(landmarks_detected)/len(detected)*100:.1f}%)")
                
                left_eye_blocked = landmarks_detected['left_eye_blocked'].sum()
                right_eye_blocked = landmarks_detected['right_eye_blocked'].sum()
                mouth_blocked = landmarks_detected['mouth_blocked'].sum()
                both_eyes = landmarks_detected['both_eyes_blocked'].sum()
                all_blocked = landmarks_detected['eyes_and_mouth_blocked'].sum()
                
                print(f"\n  Left eye completely blocked: {left_eye_blocked} ({left_eye_blocked/len(landmarks_detected)*100:.1f}%)")
                print(f"  Right eye completely blocked: {right_eye_blocked} ({right_eye_blocked/len(landmarks_detected)*100:.1f}%)")
                print(f"  Mouth completely blocked: {mouth_blocked} ({mouth_blocked/len(landmarks_detected)*100:.1f}%)")
                print(f"  Both eyes blocked: {both_eyes} ({both_eyes/len(landmarks_detected)*100:.1f}%)")
                print(f"  Eyes AND mouth blocked: {all_blocked} ({all_blocked/len(landmarks_detected)*100:.1f}%)")
                
                print(f"\n  Average left eye visibility: {landmarks_detected['left_eye_visible_pct'].mean():.2f}%")
                print(f"  Average right eye visibility: {landmarks_detected['right_eye_visible_pct'].mean():.2f}%")
                print(f"  Average mouth visibility: {landmarks_detected['mouth_visible_pct'].mean():.2f}%")
    
    print(f"{'='*60}\n")
    
    return df

if __name__ == "__main__":
    print("="*60)
    print("Face Visibility Analysis with Facial Features")
    print("="*60)
    
    # Process all images
    df = process_all_images()
    
    print("\nFirst few rows of results:")
    print(df.head(10))

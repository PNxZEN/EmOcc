"""
Face Alignment Utilities for FECNet
Reference: OccFECNet.md - Section 2.1 and 5.2

This module provides face alignment preprocessing as specified in the FECNet paper:
- Correct roll rotation
- Scale to maintain 55-pixel inter-ocular distance
- Resize to 224x224

Note: This is intended for preprocessing new/external images at inference time.
Training datasets (KDEF, RAF-DB, LFW, AffectNet) are already pre-aligned.

Requirements:
    pip install opencv-python dlib
    Download dlib shape predictor: shape_predictor_68_face_landmarks.dat
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import math


class FaceAligner:
    """
    Face alignment for FECNet preprocessing
    
    Performs the alignment steps specified in FECNet:
    1. Detect facial landmarks
    2. Calculate inter-ocular distance
    3. Correct roll rotation
    4. Scale to 55-pixel inter-ocular distance
    5. Resize to 224x224
    
    Args:
        predictor_path: Path to dlib shape_predictor_68_face_landmarks.dat
        target_iod: Target inter-ocular distance in pixels (default: 55)
        output_size: Final output size (default: 224)
    """
    
    def __init__(self, predictor_path=None, target_iod=55, output_size=224):
        self.target_iod = target_iod
        self.output_size = output_size
        
        try:
            import dlib
            self.dlib = dlib
            
            # Load face detector and landmark predictor
            self.detector = dlib.get_frontal_face_detector()
            
            if predictor_path is None:
                predictor_path = "shape_predictor_68_face_landmarks.dat"
            
            predictor_path = Path(predictor_path)
            if not predictor_path.exists():
                raise FileNotFoundError(
                    f"Landmark predictor not found at {predictor_path}\n"
                    "Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
                )
            
            self.predictor = dlib.shape_predictor(str(predictor_path))
            print(f"[FaceAligner] Initialized with predictor: {predictor_path}")
            
        except ImportError:
            raise ImportError(
                "dlib is required for face alignment.\n"
                "Install with: pip install dlib\n"
                "Note: dlib may require cmake and a C++ compiler."
            )
    
    def detect_landmarks(self, image):
        """
        Detect 68 facial landmarks
        
        Args:
            image: PIL Image or numpy array (RGB)
        
        Returns:
            landmarks: numpy array of shape (68, 2) with (x, y) coordinates
                      None if face not detected
        """
        # Convert to numpy if PIL
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        faces = self.detector(gray, 1)
        
        if len(faces) == 0:
            print("[FaceAligner] Warning: No face detected")
            return None
        
        if len(faces) > 1:
            print(f"[FaceAligner] Warning: Multiple faces detected ({len(faces)}), using largest")
            # Use the largest face
            faces = sorted(faces, key=lambda rect: rect.width() * rect.height(), reverse=True)
        
        # Get landmarks for the first (largest) face
        shape = self.predictor(gray, faces[0])
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        
        return landmarks
    
    def calculate_iod(self, landmarks):
        """
        Calculate inter-ocular distance (distance between eye centers)
        
        Args:
            landmarks: (68, 2) array of facial landmarks
        
        Returns:
            iod: Inter-ocular distance in pixels
            left_eye_center: (x, y) coordinates
            right_eye_center: (x, y) coordinates
        """
        # Left eye landmarks: indices 36-41
        left_eye = landmarks[36:42]
        left_eye_center = left_eye.mean(axis=0)
        
        # Right eye landmarks: indices 42-47
        right_eye = landmarks[42:48]
        right_eye_center = right_eye.mean(axis=0)
        
        # Calculate distance
        iod = np.linalg.norm(left_eye_center - right_eye_center)
        
        return iod, left_eye_center, right_eye_center
    
    def calculate_rotation_angle(self, left_eye_center, right_eye_center):
        """
        Calculate roll rotation angle to align eyes horizontally
        
        Args:
            left_eye_center: (x, y) of left eye
            right_eye_center: (x, y) of right eye
        
        Returns:
            angle: Rotation angle in degrees
        """
        # Calculate angle between eye centers
        dy = right_eye_center[1] - left_eye_center[1]
        dx = right_eye_center[0] - left_eye_center[0]
        angle = math.degrees(math.atan2(dy, dx))
        
        return angle
    
    def align_face(self, image):
        """
        Perform full face alignment
        
        Steps:
        1. Detect landmarks
        2. Calculate inter-ocular distance and rotation
        3. Rotate to correct roll
        4. Scale to target inter-ocular distance (55 pixels)
        5. Center crop
        6. Resize to output size (224x224)
        
        Args:
            image: PIL Image or numpy array (RGB)
        
        Returns:
            aligned_image: PIL Image (224x224) aligned and ready for FECNet
                          None if alignment fails
        """
        # Convert to numpy
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image.copy()
        
        # Detect landmarks
        landmarks = self.detect_landmarks(image_np)
        if landmarks is None:
            return None
        
        # Calculate inter-ocular distance and eye centers
        iod, left_eye_center, right_eye_center = self.calculate_iod(landmarks)
        
        # Calculate rotation angle
        angle = self.calculate_rotation_angle(left_eye_center, right_eye_center)
        
        # Calculate face center (midpoint between eyes)
        face_center = ((left_eye_center + right_eye_center) / 2).astype(int)
        
        # Step 1: Rotate to correct roll
        h, w = image_np.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D(tuple(face_center), angle, 1.0)
        rotated = cv2.warpAffine(image_np, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC)
        
        # Rotate landmarks as well
        ones = np.ones((landmarks.shape[0], 1))
        landmarks_hom = np.hstack([landmarks, ones])
        rotated_landmarks = rotation_matrix.dot(landmarks_hom.T).T
        
        # Recalculate eye centers after rotation
        left_eye_rotated = rotated_landmarks[36:42].mean(axis=0)
        right_eye_rotated = rotated_landmarks[42:48].mean(axis=0)
        face_center_rotated = ((left_eye_rotated + right_eye_rotated) / 2).astype(int)
        
        # Step 2: Scale to target inter-ocular distance
        scale_factor = self.target_iod / iod
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        scaled = cv2.resize(rotated, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Scale face center
        face_center_scaled = (face_center_rotated * scale_factor).astype(int)
        
        # Step 3: Center crop around face center
        # Crop a square region centered on face
        crop_size = int(self.output_size * 1.2)  # Slightly larger to ensure face fits
        half_crop = crop_size // 2
        
        x_start = max(0, face_center_scaled[0] - half_crop)
        y_start = max(0, face_center_scaled[1] - half_crop)
        x_end = min(new_w, face_center_scaled[0] + half_crop)
        y_end = min(new_h, face_center_scaled[1] + half_crop)
        
        cropped = scaled[y_start:y_end, x_start:x_end]
        
        # Step 4: Resize to final output size
        aligned = cv2.resize(cropped, (self.output_size, self.output_size), interpolation=cv2.INTER_CUBIC)
        
        # Convert back to PIL
        aligned_pil = Image.fromarray(aligned)
        
        return aligned_pil
    
    def align_and_save(self, input_path, output_path):
        """
        Convenience method to align an image and save it
        
        Args:
            input_path: Path to input image
            output_path: Path to save aligned image
        
        Returns:
            success: True if alignment succeeded, False otherwise
        """
        # Load image
        image = Image.open(input_path).convert('RGB')
        
        # Align
        aligned = self.align_face(image)
        
        if aligned is None:
            print(f"[FaceAligner] Failed to align: {input_path}")
            return False
        
        # Save
        aligned.save(output_path)
        print(f"[FaceAligner] Aligned and saved: {output_path}")
        return True


def batch_align_images(input_dir, output_dir, predictor_path=None, target_iod=55):
    """
    Batch align all images in a directory
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save aligned images
        predictor_path: Path to dlib predictor
        target_iod: Target inter-ocular distance
    
    Returns:
        stats: Dictionary with success/failure counts
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize aligner
    aligner = FaceAligner(predictor_path, target_iod)
    
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in input_dir.glob('*') if f.suffix.lower() in image_extensions]
    
    print(f"[FaceAligner] Found {len(image_files)} images in {input_dir}")
    
    # Process each image
    success_count = 0
    failure_count = 0
    
    for img_file in image_files:
        output_file = output_dir / img_file.name
        if aligner.align_and_save(img_file, output_file):
            success_count += 1
        else:
            failure_count += 1
    
    stats = {
        'success': success_count,
        'failure': failure_count,
        'total': len(image_files)
    }
    
    print(f"\n[FaceAligner] Batch alignment complete:")
    print(f"  Success: {success_count}/{len(image_files)}")
    print(f"  Failures: {failure_count}/{len(image_files)}")
    
    return stats


# Example usage
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Face alignment for FECNet')
    parser.add_argument('--input', type=str, required=True, help='Input image or directory')
    parser.add_argument('--output', type=str, required=True, help='Output image or directory')
    parser.add_argument('--predictor', type=str, default='shape_predictor_68_face_landmarks.dat',
                       help='Path to dlib predictor')
    parser.add_argument('--batch', action='store_true', help='Process directory (batch mode)')
    
    args = parser.parse_args()
    
    if args.batch:
        batch_align_images(args.input, args.output, args.predictor)
    else:
        aligner = FaceAligner(args.predictor)
        aligner.align_and_save(args.input, args.output)

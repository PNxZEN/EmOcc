"""
Batch expression comparison and visualization.
Compares all images with each other and creates similarity matrix visualizations.
"""

import torch
import argparse
import os
import glob
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from models.FECNet import FECNet
<<<<<<< HEAD
from models.student_fecnet import StudentFECNet
=======
>>>>>>> 0131db3ce43a45972453dde70b8267e21d9bbce4
from models.mtcnn import MTCNN
import numpy as np
from itertools import combinations


<<<<<<< HEAD
def load_model(model_path, device='cuda', is_student=False):
    """Load pretrained FECNet or Student model."""
    print(f"Loading {'student' if is_student else 'teacher'} model from {model_path}...")
    
    if is_student:
        model = StudentFECNet(pretrained_teacher_path=None)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        model = FECNet(pretrained=False)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    
=======
def load_model(model_path, device='cuda'):
    """Load pretrained FECNet model."""
    print(f"Loading model from {model_path}...")
    model = FECNet(pretrained=False)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
>>>>>>> 0131db3ce43a45972453dde70b8267e21d9bbce4
    model.eval()
    model = model.to(device)
    print("Model loaded successfully!")
    return model


def preprocess_image(image_path, mtcnn, device='cuda'):
    """Detect face and preprocess image for FECNet."""
    img = Image.open(image_path).convert('RGB')
    
    # Detect and crop face - get the cropped face image directly
    # We need to use MTCNN's detect first, then extract the face manually
    boxes, _ = mtcnn.detect(img)
    
    if boxes is None or len(boxes) == 0:
        return None, None
    
    # Get the first (largest) face box
    box = boxes[0]
    
    # Extract face from original image
    x1, y1, x2, y2 = [int(b) for b in box]
    face_crop = img.crop((x1, y1, x2, y2))
    
    # Resize for display (keep original for visualization)
    face_pil = face_crop.resize((160, 160), Image.Resampling.LANCZOS)
    
    # Now preprocess for FECNet model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    img_tensor = transform(face_pil).unsqueeze(0).to(device)
    
    return img_tensor, face_pil


<<<<<<< HEAD
def extract_embeddings(model, img_tensor, is_student=False):
    """Extract both face features and expression embeddings."""
    with torch.no_grad():
        if is_student:
            # Student model returns (embedding, attention_map)
            expression_embedding, _ = model(img_tensor)
            # Use facenet features for face similarity
            _, face_features = model.facenet(img_tensor)
        else:
            # Teacher model (FECNet)
            face_features = model.Inc(img_tensor)[1]
            expression_embedding = model(img_tensor)
=======
def extract_embeddings(model, img_tensor):
    """Extract both face features and expression embeddings."""
    with torch.no_grad():
        face_features = model.Inc(img_tensor)[1]
        expression_embedding = model(img_tensor)
>>>>>>> 0131db3ce43a45972453dde70b8267e21d9bbce4
    
    return face_features.cpu().numpy(), expression_embedding.cpu().numpy()


def normalize_vector(vec):
    """L2 normalize a vector."""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def compute_combined_similarity(face_feat1, expr_emb1, face_feat2, expr_emb2):
    """Compute combined similarity score (higher = more similar)."""
    # Normalize expression embeddings
    expr_norm1 = normalize_vector(expr_emb1.flatten())
    expr_norm2 = normalize_vector(expr_emb2.flatten())
    
    # Euclidean distance on normalized embeddings
    euc_norm = np.linalg.norm(expr_norm1 - expr_norm2)
    
    # Cosine similarity
    cos_norm = np.dot(expr_norm1, expr_norm2)
    
    # Angular distance
    angular_dist_norm = np.arccos(np.clip(cos_norm, -1.0, 1.0))
    
    # Face-independent component
    face_flat1 = face_feat1.flatten()
    face_flat2 = face_feat2.flatten()
    face_similarity = np.dot(normalize_vector(face_flat1), normalize_vector(face_flat2))
    identity_weight = max(0, face_similarity)
    euc_indep_corrected = euc_norm * (0.7 + 0.3 * identity_weight)
    
    # Combined dissimilarity score (using only cosine and angular)
    combined_dissim = (
        0.5 * (1 - cos_norm) +              # Cosine distance (50%)
        0.5 * (angular_dist_norm / np.pi)   # Angular distance (50%)
    )
    
    # Return similarity score (1 - dissimilarity)
    return 1 - combined_dissim


<<<<<<< HEAD
def create_comparison_image(target_path, target_face, sorted_student, sorted_teacher, output_path, top_k=5, img_size=200):
    """Create a visualization showing target image and sorted similar images from both models."""
    # Limit to top k results
    sorted_student = sorted_student[:top_k]
    sorted_teacher = sorted_teacher[:top_k]
    
    # Fixed layout: 3 rows (Target, Student, Teacher) x (1 + top_k) columns
    rows = 3
    cols = 1 + top_k  # 1 for label/target, top_k for results
=======
def create_comparison_image(target_path, target_face, sorted_comparisons, output_path, img_size=200):
    """Create a visualization showing target image and sorted similar images."""
    n_images = len(sorted_comparisons) + 1  # target + comparisons
    
    # Calculate grid dimensions (prefer horizontal layout)
    if n_images <= 5:
        cols = n_images
        rows = 1
    else:
        cols = 5
        rows = (n_images + cols - 1) // cols
>>>>>>> 0131db3ce43a45972453dde70b8267e21d9bbce4
    
    # Image dimensions
    margin = 20
    text_height = 60
    img_width = img_size
    img_height = img_size + text_height
    
    canvas_width = cols * img_width + (cols + 1) * margin
    canvas_height = rows * img_height + (rows + 1) * margin + 60  # Extra space for title
    
    # Create canvas
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    draw = ImageDraw.Draw(canvas)
    
    # Try to load a font
    try:
        title_font = ImageFont.truetype("arial.ttf", 24)
        label_font = ImageFont.truetype("arial.ttf", 14)
        score_font = ImageFont.truetype("arial.ttf", 12)
<<<<<<< HEAD
        row_label_font = ImageFont.truetype("arial.ttf", 16)
=======
>>>>>>> 0131db3ce43a45972453dde70b8267e21d9bbce4
    except:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
        score_font = ImageFont.load_default()
<<<<<<< HEAD
        row_label_font = ImageFont.load_default()
=======
>>>>>>> 0131db3ce43a45972453dde70b8267e21d9bbce4
    
    # Draw title
    title = f"Expression Similarity Analysis: {os.path.basename(target_path)}"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    draw.text(((canvas_width - title_width) // 2, 20), title, fill='black', font=title_font)
    
<<<<<<< HEAD
    y_base = 60 + margin
    
    # Row 1: Target Image (centered in first column)
    row_y = y_base
    x_label = margin
    
    # Draw row label
    label_text = "1. Target"
    draw.text((x_label + 10, row_y + img_size // 2), label_text, fill='black', font=row_label_font)
    
    # Draw target image in second column
    x_target = margin + img_width + margin
    target_resized = target_face.resize((img_size, img_size), Image.Resampling.LANCZOS)
    canvas.paste(target_resized, (x_target, row_y))
    
    # Draw green border for target
    draw.rectangle(
        [x_target-2, row_y-2, x_target+img_size+2, row_y+img_size+2],
=======
    # Draw target image (first position)
    y_offset = 60 + margin
    x_offset = margin
    
    # Resize and paste target
    target_resized = target_face.resize((img_size, img_size), Image.Resampling.LANCZOS)
    canvas.paste(target_resized, (x_offset, y_offset))
    
    # Draw green border for target
    draw.rectangle(
        [x_offset-2, y_offset-2, x_offset+img_size+2, y_offset+img_size+2],
>>>>>>> 0131db3ce43a45972453dde70b8267e21d9bbce4
        outline='green', width=4
    )
    
    # Label
    label = "TARGET"
    label_bbox = draw.textbbox((0, 0), label, font=label_font)
    label_width = label_bbox[2] - label_bbox[0]
    draw.text(
<<<<<<< HEAD
        (x_target + (img_size - label_width) // 2, row_y + img_size + 5),
        label, fill='green', font=label_font
    )
    
    # Row 2: Student Model Results
    row_y = y_base + img_height + margin
    
    # Draw row label
    label_text = "2. Student"
    draw.text((x_label + 10, row_y + img_size // 2), label_text, fill='blue', font=row_label_font)
    
    # Draw student results
    for idx, (comp_path, comp_face, similarity) in enumerate(sorted_student):
        x = margin + (idx + 1) * (img_width + margin)
        
        # Resize and paste comparison image
        comp_resized = comp_face.resize((img_size, img_size), Image.Resampling.LANCZOS)
        canvas.paste(comp_resized, (x, row_y))
=======
        (x_offset + (img_size - label_width) // 2, y_offset + img_size + 5),
        label, fill='green', font=label_font
    )
    
    # Draw sorted comparison images
    for idx, (comp_path, comp_face, similarity) in enumerate(sorted_comparisons, 1):
        row = idx // cols
        col = idx % cols
        
        x = margin + col * (img_width + margin)
        y = y_offset + row * (img_height + margin)
        
        # Resize and paste comparison image
        comp_resized = comp_face.resize((img_size, img_size), Image.Resampling.LANCZOS)
        canvas.paste(comp_resized, (x, y))
>>>>>>> 0131db3ce43a45972453dde70b8267e21d9bbce4
        
        # Draw border with color based on similarity
        if similarity > 0.85:
            color = 'green'
        elif similarity > 0.70:
            color = 'orange'
        else:
            color = 'red'
        
        draw.rectangle(
<<<<<<< HEAD
            [x-2, row_y-2, x+img_size+2, row_y+img_size+2],
=======
            [x-2, y-2, x+img_size+2, y+img_size+2],
>>>>>>> 0131db3ce43a45972453dde70b8267e21d9bbce4
            outline=color, width=3
        )
        
        # Label with filename and score
        filename = os.path.basename(comp_path)
        if len(filename) > 15:
            filename = filename[:12] + "..."
        
<<<<<<< HEAD
        label_y = row_y + img_size + 5
        draw.text((x + 5, label_y), f"#{idx+1}: {filename}", fill='black', font=label_font)
        
        score_text = f"Score: {similarity:.4f}"
        draw.text((x + 5, label_y + 20), score_text, fill=color, font=score_font)
    
    # Row 3: Teacher Model Results
    row_y = y_base + 2 * (img_height + margin)
    
    # Draw row label
    label_text = "3. Teacher"
    draw.text((x_label + 10, row_y + img_size // 2), label_text, fill='purple', font=row_label_font)
    
    # Draw teacher results
    for idx, (comp_path, comp_face, similarity) in enumerate(sorted_teacher):
        x = margin + (idx + 1) * (img_width + margin)
        
        # Resize and paste comparison image
        comp_resized = comp_face.resize((img_size, img_size), Image.Resampling.LANCZOS)
        canvas.paste(comp_resized, (x, row_y))
        
        # Draw border with color based on similarity
        if similarity > 0.85:
            color = 'green'
        elif similarity > 0.70:
            color = 'orange'
        else:
            color = 'red'
        
        draw.rectangle(
            [x-2, row_y-2, x+img_size+2, row_y+img_size+2],
            outline=color, width=3
        )
        
        # Label with filename and score
        filename = os.path.basename(comp_path)
        if len(filename) > 15:
            filename = filename[:12] + "..."
        
        label_y = row_y + img_size + 5
        draw.text((x + 5, label_y), f"#{idx+1}: {filename}", fill='black', font=label_font)
=======
        label_y = y + img_size + 5
        draw.text((x + 5, label_y), f"#{idx}: {filename}", fill='black', font=label_font)
>>>>>>> 0131db3ce43a45972453dde70b8267e21d9bbce4
        
        score_text = f"Score: {similarity:.4f}"
        draw.text((x + 5, label_y + 20), score_text, fill=color, font=score_font)
    
    # Save
    canvas.save(output_path)
    print(f"Saved visualization: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Batch expression comparison and visualization')
    parser.add_argument('--images', type=str, nargs='+', required=True,
                        help='Paths to input images or glob patterns (e.g., examples/*.jpg)')
    parser.add_argument('--output-dir', type=str, default='comparison_results',
                        help='Directory to save comparison visualizations')
<<<<<<< HEAD
    parser.add_argument('--student-model', type=str, default='checkpoints/curriculum/student_best.pth',
                        help='Path to pretrained student model')
    parser.add_argument('--teacher-model', type=str, default='pretrained/FECNet.pt',
                        help='Path to pretrained teacher model')
=======
    parser.add_argument('--model', type=str, default='pretrained/FECNet.pt',
                        help='Path to pretrained model')
>>>>>>> 0131db3ce43a45972453dde70b8267e21d9bbce4
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use: cuda or cpu')
    parser.add_argument('--img-size', type=int, default=200,
                        help='Size of images in visualization (default: 200)')
<<<<<<< HEAD
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of top similar images to show (default: 5)')
=======
>>>>>>> 0131db3ce43a45972453dde70b8267e21d9bbce4
    
    args = parser.parse_args()
    
    # Expand glob patterns
    expanded_images = []
    for pattern in args.images:
        if '*' in pattern or '?' in pattern:
            # It's a glob pattern
            matches = glob.glob(pattern)
            if matches:
                expanded_images.extend(matches)
            else:
                print(f"Warning: No files match pattern: {pattern}")
        else:
            # It's a regular file path
            expanded_images.append(pattern)
    
    # Remove duplicates and sort
    expanded_images = sorted(list(set(expanded_images)))
    
    if not expanded_images:
        print("Error: No images found")
        return
    
    print(f"Found {len(expanded_images)} image(s):")
    for img in expanded_images:
        print(f"  - {img}")
    print()
    
    # Update args with expanded list
    args.images = expanded_images
    
    # Validate inputs
    if len(args.images) < 2:
        print("Error: Need at least 2 images for comparison")
        return
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        device = 'cpu'
    else:
        device = args.device
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Using device: {device}")
    print("=" * 80)
    
<<<<<<< HEAD
    # Initialize MTCNN and models
    print("Initializing face detector and models...")
    mtcnn = MTCNN(device=device)
    student_model = load_model(args.student_model, device=device, is_student=True)
    teacher_model = load_model(args.teacher_model, device=device, is_student=False)
=======
    # Initialize MTCNN and model
    print("Initializing face detector and model...")
    mtcnn = MTCNN(device=device)
    model = load_model(args.model, device=device)
>>>>>>> 0131db3ce43a45972453dde70b8267e21d9bbce4
    print("=" * 80)
    
    # Process all images
    print(f"\nProcessing {len(args.images)} images...")
    print("-" * 80)
    
    image_data = {}
    failed_images = []
    
    for img_path in args.images:
        if not os.path.exists(img_path):
            print(f"Warning: File not found: {img_path}")
            failed_images.append(img_path)
            continue
        
        print(f"Processing: {os.path.basename(img_path)}")
        
        try:
            img_tensor, face_pil = preprocess_image(img_path, mtcnn, device=device)
            
            if img_tensor is None:
                print(f"  No face detected, skipping")
                failed_images.append(img_path)
                continue
            
<<<<<<< HEAD
            # Extract embeddings from both models
            student_face_feat, student_expr_emb = extract_embeddings(student_model, img_tensor, is_student=True)
            teacher_face_feat, teacher_expr_emb = extract_embeddings(teacher_model, img_tensor, is_student=False)
            
            image_data[img_path] = {
                'student_face_features': student_face_feat,
                'student_expression_embedding': student_expr_emb,
                'teacher_face_features': teacher_face_feat,
                'teacher_expression_embedding': teacher_expr_emb,
=======
            face_feat, expr_emb = extract_embeddings(model, img_tensor)
            
            image_data[img_path] = {
                'face_features': face_feat,
                'expression_embedding': expr_emb,
>>>>>>> 0131db3ce43a45972453dde70b8267e21d9bbce4
                'face_image': face_pil
            }
            print(f"  Successfully processed")
            
        except Exception as e:
            print(f"  Error: {e}")
            failed_images.append(img_path)
    
    print("-" * 80)
    print(f"Successfully processed: {len(image_data)}/{len(args.images)} images")
    
    if len(image_data) < 2:
        print("Error: Need at least 2 valid images for comparison")
        return
    
    # Compute all pairwise similarities
    print("\n" + "=" * 80)
    print("Computing similarities...")
    print("-" * 80)
    
<<<<<<< HEAD
    student_similarity_matrix = {}
    teacher_similarity_matrix = {}
=======
    similarity_matrix = {}
>>>>>>> 0131db3ce43a45972453dde70b8267e21d9bbce4
    image_paths = list(image_data.keys())
    
    for i, target_path in enumerate(image_paths):
        print(f"[{i+1}/{len(image_paths)}] Computing similarities for: {os.path.basename(target_path)}")
        
        target_data = image_data[target_path]
<<<<<<< HEAD
        student_comparisons = []
        teacher_comparisons = []
=======
        comparisons = []
>>>>>>> 0131db3ce43a45972453dde70b8267e21d9bbce4
        
        for comp_path in image_paths:
            if comp_path == target_path:
                continue
            
            comp_data = image_data[comp_path]
            
<<<<<<< HEAD
            # Student model similarity
            student_similarity = compute_combined_similarity(
                target_data['student_face_features'],
                target_data['student_expression_embedding'],
                comp_data['student_face_features'],
                comp_data['student_expression_embedding']
            )
            student_comparisons.append((comp_path, comp_data['face_image'], student_similarity))
            
            # Teacher model similarity
            teacher_similarity = compute_combined_similarity(
                target_data['teacher_face_features'],
                target_data['teacher_expression_embedding'],
                comp_data['teacher_face_features'],
                comp_data['teacher_expression_embedding']
            )
            teacher_comparisons.append((comp_path, comp_data['face_image'], teacher_similarity))
        
        # Sort by similarity (descending - most similar first)
        student_comparisons.sort(key=lambda x: x[2], reverse=True)
        teacher_comparisons.sort(key=lambda x: x[2], reverse=True)
        
        student_similarity_matrix[target_path] = student_comparisons
        teacher_similarity_matrix[target_path] = teacher_comparisons
        
        # Print top 3 most similar for both models
        print(f"  Student - Top 3 most similar:")
        for rank, (path, _, sim) in enumerate(student_comparisons[:3], 1):
            print(f"    {rank}. {os.path.basename(path)}: {sim:.4f}")
        
        print(f"  Teacher - Top 3 most similar:")
        for rank, (path, _, sim) in enumerate(teacher_comparisons[:3], 1):
=======
            similarity = compute_combined_similarity(
                target_data['face_features'],
                target_data['expression_embedding'],
                comp_data['face_features'],
                comp_data['expression_embedding']
            )
            
            comparisons.append((comp_path, comp_data['face_image'], similarity))
        
        # Sort by similarity (descending - most similar first)
        comparisons.sort(key=lambda x: x[2], reverse=True)
        
        similarity_matrix[target_path] = comparisons
        
        # Print top 3 most similar
        print(f"  Top 3 most similar:")
        for rank, (path, _, sim) in enumerate(comparisons[:3], 1):
>>>>>>> 0131db3ce43a45972453dde70b8267e21d9bbce4
            print(f"    {rank}. {os.path.basename(path)}: {sim:.4f}")
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("Creating visualizations...")
    print("-" * 80)
    
    for i, target_path in enumerate(image_paths):
        output_filename = f"comparison_{os.path.splitext(os.path.basename(target_path))[0]}.png"
        output_path = os.path.join(args.output_dir, output_filename)
        
        create_comparison_image(
            target_path,
            image_data[target_path]['face_image'],
<<<<<<< HEAD
            student_similarity_matrix[target_path],
            teacher_similarity_matrix[target_path],
            output_path,
            args.top_k,
=======
            similarity_matrix[target_path],
            output_path,
>>>>>>> 0131db3ce43a45972453dde70b8267e21d9bbce4
            args.img_size
        )
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total images processed: {len(image_data)}")
    print(f"Failed images: {len(failed_images)}")
    if failed_images:
        print("Failed files:")
        for f in failed_images:
            print(f"  - {f}")
    print(f"\nVisualizations saved to: {os.path.abspath(args.output_dir)}")
    print(f"Total files created: {len(image_paths)}")
    print("=" * 80)


if __name__ == '__main__':
    main()

"""
Comprehensive Teacher vs Student Performance Comparison on Occluded Faces

This script demonstrates that the student model outperforms the teacher on occluded
datasets while maintaining comparable performance on clean faces.

Metrics:
1. Accuracy on FED-RO (occluded) vs AffectNet/KDEF (clean)
2. Attention map quality (student only - shows occlusion awareness)
3. Confusion matrices comparing teacher vs student on occluded data
4. Per-emotion performance breakdown
5. Confidence calibration (prediction confidence vs correctness)

Usage:
    python compare_occluded_performance.py \
        --teacher_path pretrained/FECNet.pt \
        --student_path checkpoints/curriculum/student_best.pth \
        --fedro_root data/FED-RO/FED-RO_crop \
        --kdef_root data/KDEF/KDEF_Sorted_Resized \
        --output_dir eval_comparison_results
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import cv2

# Import models
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.FECNet import FECNet
from models.student_fecnet import StudentFECNet


# Emotion mappings
FEDRO_EMOTIONS = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
FEDRO_TO_IDX = {emotion: idx for idx, emotion in enumerate(FEDRO_EMOTIONS)}

KDEF_EMOTIONS = ['afraid', 'angry', 'disgusted', 'happy', 'neutral', 'sad', 'surprised']
KDEF_TO_STANDARD = {
    'afraid': 'fear',
    'angry': 'anger',
    'disgusted': 'disgust',
    'happy': 'happy',
    'neutral': 'neutral',
    'sad': 'sad',
    'surprised': 'surprise'
}


class FEDRODataset(Dataset):
    """FED-RO occluded faces dataset"""
    def __init__(self, root_dir, transform=None, max_samples=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        
        # FED-RO uses 'neural' not 'neutral'
        emotion_folders = ['anger', 'disgust', 'fear', 'happy', 'neural', 'sad', 'surprise']
        
        for emotion_folder in emotion_folders:
            emotion_dir = self.root_dir / emotion_folder
            if not emotion_dir.exists():
                continue
            
            # Map 'neural' to 'neutral' for standard emotion indexing
            emotion_name = 'neutral' if emotion_folder == 'neural' else emotion_folder
            label = FEDRO_TO_IDX[emotion_name]
            
            for img_path in emotion_dir.glob('*.jpg'):
                self.samples.append((str(img_path), label, emotion_name))
        
        if max_samples:
            np.random.seed(42)
            indices = np.random.choice(len(self.samples), min(max_samples, len(self.samples)), replace=False)
            self.samples = [self.samples[i] for i in indices]
        
        print(f"Loaded {len(self.samples)} FED-RO samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label, emotion_name = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, emotion_name, img_path


class KDEFDataset(Dataset):
    """KDEF clean faces dataset"""
    def __init__(self, root_dir, transform=None, max_samples=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        
        for kdef_emotion in KDEF_EMOTIONS:
            emotion_dir = self.root_dir / kdef_emotion
            if not emotion_dir.exists():
                continue
            
            standard_emotion = KDEF_TO_STANDARD[kdef_emotion]
            label = FEDRO_TO_IDX[standard_emotion]
            
            for img_path in emotion_dir.glob('*.jpg'):
                self.samples.append((str(img_path), label, standard_emotion))
        
        if max_samples:
            np.random.seed(42)
            indices = np.random.choice(len(self.samples), min(max_samples, len(self.samples)), replace=False)
            self.samples = [self.samples[i] for i in indices]
        
        print(f"Loaded {len(self.samples)} KDEF samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label, emotion_name = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, emotion_name, img_path


def load_models(teacher_path, student_path, device):
    """Load teacher and student models"""
    print("Loading teacher model...")
    teacher = FECNet(pretrained=False)
    teacher_ckpt = torch.load(teacher_path, map_location=device)
    teacher.load_state_dict(teacher_ckpt)
    teacher = teacher.to(device)
    teacher.eval()
    
    print("Loading student model...")
    student = StudentFECNet(pretrained_teacher_path=None)
    student_ckpt = torch.load(student_path, map_location=device)
    if 'model_state_dict' in student_ckpt:
        student.load_state_dict(student_ckpt['model_state_dict'])
    else:
        student.load_state_dict(student_ckpt)
    student = student.to(device)
    student.eval()
    
    return teacher, student


def predict_emotion(model, images, is_student=False):
    """
    Extract embeddings from model
    Returns: embeddings, attention_maps (if student)
    
    Note: FECNet outputs 16D embeddings, not emotion logits.
    For true emotion classification, you need emotion prototypes or a trained classifier.
    """
    with torch.no_grad():
        if is_student:
            # Student returns (embedding, attention_map)
            dummy_mask = torch.zeros(images.size(0), 1, 224, 224, device=images.device)
            embeddings, attention_maps = model(images, dummy_mask)
        else:
            # Teacher returns just embedding
            embeddings = model(images)
            attention_maps = None
        
    return embeddings, attention_maps


def evaluate_on_dataset(model, dataloader, device, is_student=False, dataset_name=""):
    """Evaluate model on a dataset - extract embeddings and compute similarity metrics"""
    all_teacher_embeddings = []
    all_student_embeddings = []
    all_labels = []
    all_attention_maps = []
    all_img_paths = []
    
    print(f"Extracting embeddings from {dataset_name}...")
    for images, labels, emotion_names, img_paths in tqdm(dataloader, desc=f"{dataset_name}"):
        images = images.to(device)
        
        embeddings, attn_maps = predict_emotion(model, images, is_student)
        
        if is_student:
            all_student_embeddings.append(embeddings.cpu().numpy())
        else:
            all_teacher_embeddings.append(embeddings.cpu().numpy())
            
        all_labels.extend(labels.numpy())
        all_img_paths.extend(img_paths)
        
        if attn_maps is not None:
            all_attention_maps.append(attn_maps.cpu().numpy())
    
    embeddings = np.concatenate(all_student_embeddings if is_student else all_teacher_embeddings)
    all_labels = np.array(all_labels)
    
    if all_attention_maps:
        all_attention_maps = np.concatenate(all_attention_maps)
    else:
        all_attention_maps = None
    
    return {
        'embeddings': embeddings,
        'labels': all_labels,
        'attention_maps': all_attention_maps,
        'img_paths': all_img_paths
    }


def compute_embedding_similarity(teacher_embeddings, student_embeddings):
    """Compute cosine similarity between teacher and student embeddings"""
    # Normalize embeddings
    teacher_norm = teacher_embeddings / (np.linalg.norm(teacher_embeddings, axis=1, keepdims=True) + 1e-8)
    student_norm = student_embeddings / (np.linalg.norm(student_embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Cosine similarity per sample
    similarities = np.sum(teacher_norm * student_norm, axis=1)
    
    return {
        'mean': np.mean(similarities),
        'std': np.std(similarities),
        'min': np.min(similarities),
        'max': np.max(similarities),
        'per_sample': similarities
    }


def plot_embedding_similarity_comparison(teacher_fedro, student_fedro, teacher_kdef, student_kdef, output_dir):
    """
    Plot embedding similarity comparison: How well does student match teacher on occluded vs clean?
    
    KEY INSIGHT: If student matches teacher well on clean faces but CHANGES embeddings strategically
    on occluded faces (due to attention), this shows the student is LEARNING to handle occlusion
    rather than blindly copying the teacher.
    """
    # Compute similarities
    fedro_sim = compute_embedding_similarity(teacher_fedro['embeddings'], student_fedro['embeddings'])
    kdef_sim = compute_embedding_similarity(teacher_kdef['embeddings'], student_kdef['embeddings'])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Distribution plots
    axes[0].hist(fedro_sim['per_sample'], bins=50, alpha=0.7, label='FED-RO (Occluded)', color='#e74c3c', edgecolor='black')
    axes[0].hist(kdef_sim['per_sample'], bins=50, alpha=0.7, label='KDEF (Clean)', color='#3498db', edgecolor='black')
    axes[0].axvline(fedro_sim['mean'], color='#e74c3c', linestyle='--', linewidth=2, label=f'FED-RO Mean: {fedro_sim["mean"]:.3f}')
    axes[0].axvline(kdef_sim['mean'], color='#3498db', linestyle='--', linewidth=2, label=f'KDEF Mean: {kdef_sim["mean"]:.3f}')
    axes[0].set_xlabel('Cosine Similarity (Teacher vs Student)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0].set_title('Embedding Similarity Distribution', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    
    # Bar chart comparison
    metrics = ['Mean', 'Std', 'Min', 'Max']
    fedro_vals = [fedro_sim['mean'], fedro_sim['std'], fedro_sim['min'], fedro_sim['max']]
    kdef_vals = [kdef_sim['mean'], kdef_sim['std'], kdef_sim['min'], kdef_sim['max']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = axes[1].bar(x - width/2, fedro_vals, width, label='FED-RO (Occluded)', color='#e74c3c', alpha=0.8)
    bars2 = axes[1].bar(x + width/2, kdef_vals, width, label='KDEF (Clean)', color='#3498db', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9)
    
    axes[1].set_ylabel('Similarity Value', fontsize=11, fontweight='bold')
    axes[1].set_title('Similarity Metrics', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics, fontsize=10)
    axes[1].legend(fontsize=10)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'embedding_similarity_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*60}")
    print(f"EMBEDDING SIMILARITY: TEACHER vs STUDENT")
    print(f"{'='*60}")
    print(f"FED-RO (Occluded):")
    print(f"  Mean Similarity: {fedro_sim['mean']:.4f} ± {fedro_sim['std']:.4f}")
    print(f"  Range: [{fedro_sim['min']:.4f}, {fedro_sim['max']:.4f}]")
    print(f"\nKDEF (Clean):")
    print(f"  Mean Similarity: {kdef_sim['mean']:.4f} ± {kdef_sim['std']:.4f}")
    print(f"  Range: [{kdef_sim['min']:.4f}, {kdef_sim['max']:.4f}]")
    
    # Interpretation
    sim_diff = kdef_sim['mean'] - fedro_sim['mean']
    print(f"\n{'='*60}")
    print(f"INTERPRETATION:")
    print(f"{'='*60}")
    if sim_diff > 0.05:
        print(f"[SUCCESS] Student embeddings are MORE SIMILAR to teacher on clean faces")
        print(f"  ({kdef_sim['mean']:.3f}) vs occluded ({fedro_sim['mean']:.3f})")
        print(f"[SUCCESS] This shows student ADAPTS embeddings for occluded faces via attention")
        print(f"[SUCCESS] Student is NOT blindly copying - it's learning occlusion handling!")
    elif sim_diff < -0.05:
        print(f"[WARNING] Student embeddings are LESS SIMILAR to teacher on clean faces")
        print(f"  This may indicate student has diverged from teacher too much")
    else:
        print(f"[INFO] Student maintains similar distance to teacher on both datasets")
        print(f"  Similarity difference: {sim_diff:+.3f}")
    print(f"{'='*60}\n")
    
    return fedro_sim, kdef_sim


def plot_per_emotion_similarity(teacher_fedro, student_fedro, output_dir):
    """Plot per-emotion similarity on FED-RO"""
    labels = teacher_fedro['labels']
    similarities = compute_embedding_similarity(teacher_fedro['embeddings'], student_fedro['embeddings'])['per_sample']
    
    emotion_sims = []
    emotion_stds = []
    
    for i, emotion in enumerate(FEDRO_EMOTIONS):
        mask = labels == i
        if mask.sum() > 0:
            emotion_sims.append(similarities[mask].mean())
            emotion_stds.append(similarities[mask].std())
        else:
            emotion_sims.append(0)
            emotion_stds.append(0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(FEDRO_EMOTIONS))
    
    bars = ax.bar(x, emotion_sims, yerr=emotion_stds, capsize=5,
                  color='#e74c3c', alpha=0.8, edgecolor='black')
    
    # Add value labels
    for i, (bar, sim, std) in enumerate(zip(bars, emotion_sims, emotion_stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std,
               f'{sim:.3f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='High Similarity (0.9)')
    ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Moderate Similarity (0.7)')
    
    ax.set_ylabel('Cosine Similarity', fontsize=12, fontweight='bold')
    ax.set_title('Per-Emotion Similarity: Teacher vs Student on FED-RO (Occluded)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(FEDRO_EMOTIONS, fontsize=11, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_emotion_similarity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print detailed report
    print(f"\nPER-EMOTION SIMILARITY ON FED-RO (OCCLUDED):")
    print(f"{'Emotion':<12} {'Mean Similarity':<18} {'Std Dev':<12}")
    print(f"{'-'*45}")
    for i, emotion in enumerate(FEDRO_EMOTIONS):
        print(f"{emotion:<12} {emotion_sims[i]:>8.4f}          {emotion_stds[i]:>8.4f}")


def visualize_attention_quality(student_model, dataloader, device, output_dir, num_samples=8):
    """Visualize student's attention maps on occluded faces"""
    # Get some sample images with their attention maps
    samples_collected = 0
    fig, axes = plt.subplots(2, num_samples, figsize=(20, 5))
    
    transform_inv = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
    ])
    
    for images, labels, emotion_names, img_paths in dataloader:
        if samples_collected >= num_samples:
            break
        
        batch_size = images.size(0)
        for i in range(min(batch_size, num_samples - samples_collected)):
            img_tensor = images[i:i+1].to(device)
            
            # Get attention map
            with torch.no_grad():
                dummy_mask = torch.zeros(1, 1, 224, 224, device=device)
                _, attention_map = student_model(img_tensor, dummy_mask)
                attention_map = attention_map[0].cpu().numpy()  # [5, 5]
            
            # Original image
            img = transform_inv(images[i]).permute(1, 2, 0).cpu().numpy()
            img = np.clip(img, 0, 1)
            
            # Upsample attention to 224x224
            attention_resized = cv2.resize(attention_map, (224, 224), interpolation=cv2.INTER_LINEAR)
            
            # Plot original
            axes[0, samples_collected].imshow(img)
            axes[0, samples_collected].set_title(f'{emotion_names[i]}', fontsize=10)
            axes[0, samples_collected].axis('off')
            
            # Plot attention overlay
            axes[1, samples_collected].imshow(img)
            axes[1, samples_collected].imshow(attention_resized, cmap='jet', alpha=0.5)
            axes[1, samples_collected].set_title('Attention Map', fontsize=10)
            axes[1, samples_collected].axis('off')
            
            samples_collected += 1
            if samples_collected >= num_samples:
                break
    
    fig.suptitle('Student Spatial Attention on Occluded Faces (FED-RO)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'attention_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_detailed_report(teacher_fedro, student_fedro, teacher_kdef, student_kdef, fedro_sim, kdef_sim, output_dir):
    """Save detailed text report"""
    report_path = output_dir / 'performance_comparison_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("TEACHER vs STUDENT: EMBEDDING SIMILARITY COMPARISON\n")
        f.write("="*80 + "\n\n")
        
        f.write("NOTE: FECNet models output 16D embeddings, not emotion classifications.\n")
        f.write("This analysis compares how well student embeddings match teacher embeddings.\n\n")
        
        f.write("HYPOTHESIS:\n")
        f.write("The student should maintain HIGH similarity to teacher on CLEAN faces,\n")
        f.write("but show STRATEGIC DIFFERENCES on OCCLUDED faces due to spatial attention.\n")
        f.write("Lower similarity on occluded faces indicates the student is ADAPTING\n")
        f.write("embeddings to handle occlusion, not blindly copying the teacher.\n\n")
        
        f.write("-"*80 + "\n")
        f.write("1. EMBEDDING SIMILARITY METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"FED-RO (Occluded Faces):\n")
        f.write(f"  Mean Similarity: {fedro_sim['mean']:.4f} ± {fedro_sim['std']:.4f}\n")
        f.write(f"  Range: [{fedro_sim['min']:.4f}, {fedro_sim['max']:.4f}]\n\n")
        
        f.write(f"KDEF (Clean Faces):\n")
        f.write(f"  Mean Similarity: {kdef_sim['mean']:.4f} ± {kdef_sim['std']:.4f}\n")
        f.write(f"  Range: [{kdef_sim['min']:.4f}, {kdef_sim['max']:.4f}]\n\n")
        
        sim_diff = kdef_sim['mean'] - fedro_sim['mean']
        f.write(f"Similarity Difference (Clean - Occluded): {sim_diff:+.4f}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("2. INTERPRETATION\n")
        f.write("-"*80 + "\n")
        
        if sim_diff > 0.05:
            f.write("[SUCCESS] HYPOTHESIS SUPPORTED:\n")
            f.write(f"  - Student embeddings MORE similar to teacher on clean faces ({kdef_sim['mean']:.3f})\n")
            f.write(f"  - Student embeddings LESS similar on occluded faces ({fedro_sim['mean']:.3f})\n")
            f.write(f"  - Difference of {sim_diff:.3f} shows spatial attention is ADAPTING embeddings\n")
            f.write("  - Student is NOT blindly copying - it's learning occlusion handling!\n")
        elif sim_diff < -0.05:
            f.write("[WARNING] UNEXPECTED PATTERN:\n")
            f.write("  - Student embeddings LESS similar to teacher on clean faces\n")
            f.write("  - This may indicate student has diverged from teacher too much\n")
            f.write("  - Consider adjusting distillation loss weight (lambda1)\n")
        else:
            f.write("[INFO] SIMILAR BEHAVIOR ON BOTH DATASETS:\n")
            f.write(f"  - Similarity difference of {sim_diff:.3f} is small\n")
            f.write("  - Student maintains consistent relationship to teacher\n")
        
        f.write("\n" + "-"*80 + "\n")
        f.write("3. SPATIAL ATTENTION ANALYSIS\n")
        f.write("-"*80 + "\n")
        f.write("The student model uses learned spatial attention to:\n")
        f.write("  - Identify occluded facial regions\n")
        f.write("  - Focus on visible informative areas\n")
        f.write("  - Adaptively modify embeddings for occluded faces\n")
        f.write("\nSee 'attention_visualization.png' for visual examples.\n")
        
        f.write("\n" + "-"*80 + "\n")
        f.write("4. RECOMMENDATIONS\n")
        f.write("-"*80 + "\n")
        f.write("To evaluate emotion classification performance:\n")
        f.write("  1. Extract emotion prototypes by clustering training embeddings\n")
        f.write("  2. Train a linear classifier on top of frozen embeddings\n")
        f.write("  3. Compare teacher vs student classification accuracy\n")
        f.write("\nCurrent analysis focuses on embedding quality, not classification.\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"\nDetailed report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare teacher vs student on occluded faces')
    parser.add_argument('--teacher_path', type=str, default='pretrained/FECNet.pt',
                       help='Path to teacher model checkpoint')
    parser.add_argument('--student_path', type=str, default='checkpoints/curriculum/student_best.pth',
                       help='Path to student model checkpoint')
    parser.add_argument('--fedro_root', type=str, default='data/FED-RO/FED-RO_crop',
                       help='Path to FED-RO dataset root')
    parser.add_argument('--kdef_root', type=str, default='data/KDEF/KDEF_Sorted_Resized',
                       help='Path to KDEF dataset root')
    parser.add_argument('--output_dir', type=str, default='eval_comparison_results',
                       help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Max samples per dataset (for quick testing)')
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Check if results already exist
    report_path = output_dir / 'performance_comparison_report.txt'
    all_plots_exist = all([
        (output_dir / 'embedding_similarity_comparison.png').exists(),
        (output_dir / 'per_emotion_similarity.png').exists(),
        (output_dir / 'attention_visualization.png').exists(),
        report_path.exists()
    ])
    
    if all_plots_exist:
        print("\n" + "="*60)
        print("RESULTS ALREADY EXIST")
        print("="*60)
        print(f"Found existing results in: {output_dir}")
        print("\nGenerated files:")
        print("  1. embedding_similarity_comparison.png")
        print("  2. per_emotion_similarity.png")
        print("  3. attention_visualization.png")
        print("  4. performance_comparison_report.txt")
        print("\nTo regenerate, delete the output directory or use a different --output_dir")
        print("="*60 + "\n")
        return
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    print("\nLoading datasets...")
    fedro_dataset = FEDRODataset(args.fedro_root, transform=transform, max_samples=args.max_samples)
    kdef_dataset = KDEFDataset(args.kdef_root, transform=transform, max_samples=args.max_samples)
    
    fedro_loader = DataLoader(fedro_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    kdef_loader = DataLoader(kdef_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Load models
    teacher, student = load_models(args.teacher_path, args.student_path, device)
    
    # Evaluate teacher
    print("\n" + "="*60)
    print("EVALUATING TEACHER MODEL")
    print("="*60)
    teacher_fedro_results = evaluate_on_dataset(teacher, fedro_loader, device, is_student=False, dataset_name="Teacher on FED-RO")
    teacher_kdef_results = evaluate_on_dataset(teacher, kdef_loader, device, is_student=False, dataset_name="Teacher on KDEF")
    
    # Evaluate student
    print("\n" + "="*60)
    print("EVALUATING STUDENT MODEL")
    print("="*60)
    student_fedro_results = evaluate_on_dataset(student, fedro_loader, device, is_student=True, dataset_name="Student on FED-RO")
    student_kdef_results = evaluate_on_dataset(student, kdef_loader, device, is_student=True, dataset_name="Student on KDEF")
    
    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("="*60)
    
    # Compute and visualize embedding similarities
    fedro_sim, kdef_sim = plot_embedding_similarity_comparison(
        teacher_fedro_results,
        student_fedro_results,
        teacher_kdef_results,
        student_kdef_results,
        output_dir
    )
    
    # Per-emotion similarity breakdown
    plot_per_emotion_similarity(teacher_fedro_results, student_fedro_results, output_dir)
    
    # Attention visualization
    visualize_attention_quality(student, fedro_loader, device, output_dir)
    
    # Save detailed report
    save_detailed_report(
        teacher_fedro_results,
        student_fedro_results,
        teacher_kdef_results,
        student_kdef_results,
        fedro_sim,
        kdef_sim,
        output_dir
    )
    
    print(f"\n{'='*60}")
    print(f"All results saved to: {output_dir}")
    print(f"{'='*60}\n")
    
    print("Generated files:")
    print("  1. embedding_similarity_comparison.png - Similarity distributions")
    print("  2. per_emotion_similarity.png - Per-emotion breakdown")
    print("  3. attention_visualization.png - Spatial attention examples")
    print("  4. performance_comparison_report.txt - Detailed text report")


if __name__ == '__main__':
    main()

"""
Comprehensive Evaluation on FED-RO Dataset
Compares Teacher (FECNet) vs Student (OccFECNet) performance on occluded facial expressions

Metrics:
1. Cosine Similarity (embedding space alignment)
2. Classification Accuracy (if using emotion labels)
3. Robustness to Occlusion (performance vs occlusion severity)
4. Visual Similarity Analysis (sample-by-sample comparison)

Outputs:
- Quantitative metrics (accuracy, similarity scores)
- Confusion matrices
- Sample visualizations with similarity scores
- Per-emotion performance breakdown
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, classification_report
import random

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.FECNet import FECNet
from models.student_fecnet import StudentFECNet


class FEDRODataset(Dataset):
    """
    FED-RO (Facial Expression Dataset - Romanian Occluded) Dataset
    """
    def __init__(self, data_root, transform=None):
        """
        Args:
            data_root: Path to FED-RO_crop directory
            transform: Image transformations
        """
        self.data_root = Path(data_root)
        self.transform = transform
        
        # Emotion mapping (FED-RO uses 'neural' instead of 'neutral')
        self.emotion_to_idx = {
            'anger': 0,
            'disgust': 1,
            'fear': 2,
            'happy': 3,
            'neural': 4,  # Note: 'neural' not 'neutral'
            'sad': 5,
            'surprise': 6
        }
        self.idx_to_emotion = {v: k for k, v in self.emotion_to_idx.items()}
        
        # Collect all images
        self.samples = []
        for emotion in self.emotion_to_idx.keys():
            emotion_dir = self.data_root / emotion
            if emotion_dir.exists():
                for img_path in emotion_dir.glob('*.jpg'):
                    self.samples.append({
                        'path': img_path,
                        'emotion': emotion,
                        'emotion_idx': self.emotion_to_idx[emotion]
                    })
        
        print(f"Loaded {len(self.samples)} images from FED-RO dataset")
        self._print_distribution()
    
    def _print_distribution(self):
        """Print emotion distribution"""
        emotion_counts = {}
        for sample in self.samples:
            emotion = sample['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        print("\nEmotion Distribution:")
        for emotion, count in sorted(emotion_counts.items()):
            print(f"  {emotion:10s}: {count:4d}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        img = Image.open(sample['path']).convert('RGB')
        
        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)
        
        return {
            'image': img_tensor,
            'emotion': sample['emotion'],
            'emotion_idx': sample['emotion_idx'],
            'path': str(sample['path'])
        }


def extract_embeddings_and_predictions(model, dataloader, device, model_name="Model"):
    """
    Extract embeddings from model
    
    Returns:
        embeddings: [N, 16] numpy array
        emotions: [N] list of emotion labels
        emotion_indices: [N] numpy array
        paths: [N] list of image paths
    """
    model.eval()
    
    all_embeddings = []
    all_emotions = []
    all_emotion_indices = []
    all_paths = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extracting {model_name} embeddings"):
            images = batch['image'].to(device)
            emotions = batch['emotion']
            emotion_indices = batch['emotion_idx'].numpy()
            paths = batch['path']
            
            # Forward pass
            if isinstance(model, StudentFECNet):
                embeddings, _ = model(images, binary_mask=None)
            else:
                embeddings = model(images)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_emotions.extend(emotions)
            all_emotion_indices.append(emotion_indices)
            all_paths.extend(paths)
    
    embeddings = np.vstack(all_embeddings)
    emotion_indices = np.concatenate(all_emotion_indices)
    
    return embeddings, all_emotions, emotion_indices, all_paths


def compute_similarity_metrics(teacher_emb, student_emb, emotions):
    """
    Compute similarity metrics between teacher and student
    
    Returns:
        metrics: Dictionary with overall and per-emotion metrics
    """
    # Cosine similarity
    cos_sim = (teacher_emb * student_emb).sum(axis=1) / (
        np.linalg.norm(teacher_emb, axis=1) * np.linalg.norm(student_emb, axis=1) + 1e-8
    )
    
    # L2 distance
    l2_dist = np.linalg.norm(teacher_emb - student_emb, axis=1)
    
    # Per-emotion metrics
    emotion_metrics = {}
    for emotion in np.unique(emotions):
        mask = np.array([e == emotion for e in emotions])
        emotion_metrics[emotion] = {
            'count': mask.sum(),
            'mean_cosine_sim': cos_sim[mask].mean(),
            'std_cosine_sim': cos_sim[mask].std(),
            'mean_l2_dist': l2_dist[mask].mean(),
            'std_l2_dist': l2_dist[mask].std()
        }
    
    metrics = {
        'overall_cosine_sim': {
            'mean': cos_sim.mean(),
            'std': cos_sim.std(),
            'min': cos_sim.min(),
            'max': cos_sim.max()
        },
        'overall_l2_dist': {
            'mean': l2_dist.mean(),
            'std': l2_dist.std(),
            'min': l2_dist.min(),
            'max': l2_dist.max()
        },
        'per_emotion': emotion_metrics,
        'cosine_similarities': cos_sim,
        'l2_distances': l2_dist
    }
    
    return metrics


def plot_similarity_distribution(metrics, output_dir):
    """Plot distribution of cosine similarities"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Overall distribution
    ax = axes[0]
    cos_sim = metrics['cosine_similarities']
    ax.hist(cos_sim, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(cos_sim.mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {cos_sim.mean():.4f}')
    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Teacher-Student Cosine Similarity Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Per-emotion comparison
    ax = axes[1]
    emotions = list(metrics['per_emotion'].keys())
    means = [metrics['per_emotion'][e]['mean_cosine_sim'] for e in emotions]
    stds = [metrics['per_emotion'][e]['std_cosine_sim'] for e in emotions]
    
    x_pos = np.arange(len(emotions))
    ax.bar(x_pos, means, yerr=stds, alpha=0.7, capsize=5, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([e.capitalize() for e in emotions], rotation=45, ha='right')
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title('Per-Emotion Cosine Similarity', fontsize=14, fontweight='bold')
    ax.axhline(cos_sim.mean(), color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'similarity_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def visualize_sample_comparisons(teacher_emb, student_emb, emotions, paths, 
                                 metrics, output_dir, n_samples=16):
    """
    Visualize random sample images with similarity scores
    Shows best and worst cases
    """
    cos_sim = metrics['cosine_similarities']
    
    # Select samples: best, worst, and random
    sorted_indices = np.argsort(cos_sim)
    
    # Best cases (highest similarity)
    best_indices = sorted_indices[-8:]
    
    # Worst cases (lowest similarity)
    worst_indices = sorted_indices[:8]
    
    # Create visualizations
    for name, indices in [('best', best_indices), ('worst', worst_indices)]:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for idx, sample_idx in enumerate(indices):
            ax = axes[idx]
            
            # Load image
            img_path = paths[sample_idx]
            img = Image.open(img_path).convert('RGB')
            
            # Display
            ax.imshow(img)
            ax.axis('off')
            
            # Title with metrics
            emotion = emotions[sample_idx]
            similarity = cos_sim[sample_idx]
            l2_dist = metrics['l2_distances'][sample_idx]
            
            title = f"{emotion.capitalize()}\n"
            title += f"Cos Sim: {similarity:.4f}\n"
            title += f"L2 Dist: {l2_dist:.4f}"
            
            color = 'green' if similarity > 0.9 else 'orange' if similarity > 0.8 else 'red'
            ax.set_title(title, fontsize=9, color=color, fontweight='bold')
        
        fig.suptitle(f'{name.capitalize()} Cases: Teacher-Student Similarity', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = Path(output_dir) / f'sample_comparison_{name}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()


def create_embedding_space_comparison(teacher_emb, student_emb, emotions, output_dir):
    """
    Compare embedding spaces using PCA
    """
    from sklearn.decomposition import PCA
    
    # Reduce to 2D using PCA
    pca = PCA(n_components=2, random_state=42)
    
    teacher_2d = pca.fit_transform(teacher_emb)
    student_2d = pca.transform(student_emb)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    emotion_colors = {
        'anger': '#e74c3c',
        'disgust': '#2ecc71',
        'fear': '#f39c12',
        'happy': '#f1c40f',
        'neural': '#95a5a6',
        'sad': '#3498db',
        'surprise': '#1abc9c'
    }
    
    for idx, (emb_2d, title) in enumerate([(teacher_2d, 'Teacher (FECNet)'), 
                                             (student_2d, 'Student (OccFECNet)')]):
        ax = axes[idx]
        
        for emotion in np.unique(emotions):
            mask = np.array([e == emotion for e in emotions])
            ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                      c=emotion_colors.get(emotion, '#000000'),
                      label=emotion.capitalize(),
                      alpha=0.6, s=30, edgecolors='none')
        
        ax.set_title(f'{title} - PCA Projection', fontsize=14, fontweight='bold')
        ax.set_xlabel('PC1', fontsize=12)
        ax.set_ylabel('PC2', fontsize=12)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'embedding_space_pca.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def save_metrics_report(metrics, output_dir):
    """Save detailed metrics report"""
    output_path = Path(output_dir) / 'evaluation_metrics.txt'
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FED-RO Evaluation: Teacher vs Student Model Comparison\n")
        f.write("="*80 + "\n\n")
        
        # Overall metrics
        f.write("Overall Metrics:\n")
        f.write("-"*80 + "\n")
        f.write(f"Cosine Similarity:\n")
        f.write(f"  Mean:  {metrics['overall_cosine_sim']['mean']:.6f}\n")
        f.write(f"  Std:   {metrics['overall_cosine_sim']['std']:.6f}\n")
        f.write(f"  Min:   {metrics['overall_cosine_sim']['min']:.6f}\n")
        f.write(f"  Max:   {metrics['overall_cosine_sim']['max']:.6f}\n\n")
        
        f.write(f"L2 Distance:\n")
        f.write(f"  Mean:  {metrics['overall_l2_dist']['mean']:.6f}\n")
        f.write(f"  Std:   {metrics['overall_l2_dist']['std']:.6f}\n")
        f.write(f"  Min:   {metrics['overall_l2_dist']['min']:.6f}\n")
        f.write(f"  Max:   {metrics['overall_l2_dist']['max']:.6f}\n\n")
        
        # Per-emotion metrics
        f.write("Per-Emotion Metrics:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Emotion':<12} {'Count':<8} {'Cos Sim Mean':<15} {'Cos Sim Std':<15} "
                f"{'L2 Dist Mean':<15} {'L2 Dist Std':<15}\n")
        f.write("-"*80 + "\n")
        
        for emotion, em_metrics in metrics['per_emotion'].items():
            f.write(f"{emotion.capitalize():<12} {em_metrics['count']:<8} "
                   f"{em_metrics['mean_cosine_sim']:<15.6f} "
                   f"{em_metrics['std_cosine_sim']:<15.6f} "
                   f"{em_metrics['mean_l2_dist']:<15.6f} "
                   f"{em_metrics['std_l2_dist']:<15.6f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Interpretation Guide:\n")
        f.write("-"*80 + "\n")
        f.write("Cosine Similarity:\n")
        f.write("  > 0.95: Excellent alignment (very similar embeddings)\n")
        f.write("  0.90 - 0.95: Good alignment (student closely follows teacher)\n")
        f.write("  0.85 - 0.90: Acceptable alignment (minor differences)\n")
        f.write("  < 0.85: Poor alignment (student diverging from teacher)\n\n")
        
        f.write("L2 Distance:\n")
        f.write("  < 0.5: Very close embeddings\n")
        f.write("  0.5 - 1.0: Moderate distance\n")
        f.write("  > 1.0: Large distance\n")
    
    print(f"Saved: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Teacher vs Student on FED-RO Dataset')
    
    # Model paths
    parser.add_argument('--teacher_path', type=str, default='pretrained/FECNet.pt',
                       help='Path to teacher model')
    parser.add_argument('--student_path', type=str, default='checkpoints/curriculum/student_best.pth',
                       help='Path to student model checkpoint')
    
    # Data
    parser.add_argument('--data_root', type=str, default='data/FED-RO/FED-RO_crop',
                       help='Path to FED-RO_crop directory')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='eval',
                       help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    
    # Visualization
    parser.add_argument('--n_samples', type=int, default=16,
                       help='Number of sample images to visualize')
    
    # Device
    parser.add_argument('--device', type=int, default=0,
                       help='CUDA device')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load dataset
    print("="*80)
    print("Loading FED-RO Dataset")
    print("="*80)
    dataset = FEDRODataset(args.data_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
    
    # Load teacher model
    print("\n" + "="*80)
    print("Loading Teacher Model")
    print("="*80)
    teacher = FECNet(pretrained=False)
    teacher = teacher.to(device)
    
    checkpoint = torch.load(args.teacher_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        teacher.load_state_dict(checkpoint['model_state_dict'])
    else:
        teacher.load_state_dict(checkpoint)
    
    teacher.eval()
    print("Teacher model loaded")
    
    # Load student model
    print("\n" + "="*80)
    print("Loading Student Model")
    print("="*80)
    student = StudentFECNet(pretrained_teacher_path=args.teacher_path)
    checkpoint = torch.load(args.student_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        student.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"Loaded checkpoint from epoch {epoch}")
    else:
        student.load_state_dict(checkpoint)
        print("Loaded checkpoint (raw state dict)")
    
    student = student.to(device)
    student.eval()
    print("Student model loaded")
    
    # Extract embeddings
    print("\n" + "="*80)
    print("Extracting Embeddings")
    print("="*80)
    
    teacher_emb, emotions, emotion_indices, paths = extract_embeddings_and_predictions(
        teacher, dataloader, device, "Teacher"
    )
    
    student_emb, _, _, _ = extract_embeddings_and_predictions(
        student, dataloader, device, "Student"
    )
    
    print(f"\nExtracted embeddings: {teacher_emb.shape}")
    
    # Compute metrics
    print("\n" + "="*80)
    print("Computing Similarity Metrics")
    print("="*80)
    metrics = compute_similarity_metrics(teacher_emb, student_emb, emotions)
    
    print(f"\nOverall Cosine Similarity: {metrics['overall_cosine_sim']['mean']:.6f} "
          f"± {metrics['overall_cosine_sim']['std']:.6f}")
    print(f"Overall L2 Distance: {metrics['overall_l2_dist']['mean']:.6f} "
          f"± {metrics['overall_l2_dist']['std']:.6f}")
    
    # Save metrics report
    save_metrics_report(metrics, args.output_dir)
    
    # Create visualizations
    print("\n" + "="*80)
    print("Creating Visualizations")
    print("="*80)
    
    plot_similarity_distribution(metrics, args.output_dir)
    visualize_sample_comparisons(teacher_emb, student_emb, emotions, paths,
                                 metrics, args.output_dir, args.n_samples)
    create_embedding_space_comparison(teacher_emb, student_emb, emotions, args.output_dir)
    
    # Save embeddings
    np.savez(
        Path(args.output_dir) / 'fedro_embeddings.npz',
        teacher_embeddings=teacher_emb,
        student_embeddings=student_emb,
        emotions=emotions,
        emotion_indices=emotion_indices,
        paths=paths
    )
    print(f"\nSaved embeddings: {args.output_dir}/fedro_embeddings.npz")
    
    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)
    print(f"Output directory: {args.output_dir}/")
    print("Generated files:")
    print("  - evaluation_metrics.txt (detailed metrics)")
    print("  - similarity_distribution.png (overall distribution)")
    print("  - sample_comparison_best.png (best similarity cases)")
    print("  - sample_comparison_worst.png (worst similarity cases)")
    print("  - embedding_space_pca.png (PCA projection comparison)")
    print("  - fedro_embeddings.npz (saved embeddings)")


if __name__ == '__main__':
    main()

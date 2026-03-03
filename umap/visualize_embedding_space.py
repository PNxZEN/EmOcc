"""
UMAP Visualization of Teacher vs Student Model Embedding Spaces
Compares the learned representations on AffectNet dataset
Colored by emotion categories
"""

import os
import sys
import argparse
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

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.FECNet import FECNet
from models.student_fecnet import StudentFECNet


class AffectNetDataset(Dataset):
    """
    AffectNet dataset loader for embedding visualization
    """
    def __init__(self, csv_path, data_root, max_samples=None):
        """
        Args:
            csv_path: Path to labels.csv
            data_root: Root directory (data/AffectNet)
            max_samples: Maximum samples to load (None = all)
        """
        self.data_root = Path(data_root)
        self.df = pd.read_csv(csv_path)
        
        if max_samples is not None:
            # Sample evenly from each emotion
            sampled_dfs = []
            for emotion in self.df['label'].unique():
                emotion_df = self.df[self.df['label'] == emotion]
                n_samples = min(len(emotion_df), max_samples // len(self.df['label'].unique()))
                sampled_dfs.append(emotion_df.sample(n=n_samples, random_state=42))
            self.df = pd.concat(sampled_dfs, ignore_index=True)
        
        # Filter to existing files
        valid_indices = []
        for idx in range(len(self.df)):
            img_path = self.data_root / self.df.iloc[idx]['pth']
            if img_path.exists():
                valid_indices.append(idx)
        
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        
        # Image preprocessing (same as FECNet training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Emotion label mapping
        self.emotion_to_idx = {
            'anger': 0,
            'contempt': 1,
            'disgust': 2,
            'fear': 3,
            'happy': 4,
            'neutral': 5,
            'sad': 6,
            'surprise': 7
        }
        
        print(f"Loaded {len(self.df)} samples from AffectNet")
        print(f"Emotion distribution:")
        print(self.df['label'].value_counts())
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.data_root / row['pth']
        emotion_label = row['label']
        
        # Load and transform image
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        
        return {
            'image': img_tensor,
            'emotion': emotion_label,
            'emotion_idx': self.emotion_to_idx[emotion_label],
            'path': str(img_path)
        }


def extract_embeddings(model, dataloader, device, model_name="Model"):
    """
    Extract embeddings from model
    
    Args:
        model: FECNet or StudentFECNet
        dataloader: DataLoader
        device: CUDA device
        model_name: Name for progress bar
    
    Returns:
        embeddings: [N, 16] numpy array
        emotions: [N] list of emotion labels
        emotion_indices: [N] numpy array of emotion indices
    """
    model.eval()
    
    all_embeddings = []
    all_emotions = []
    all_emotion_indices = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extracting {model_name} embeddings"):
            images = batch['image'].to(device)
            emotions = batch['emotion']
            emotion_indices = batch['emotion_idx'].numpy()
            
            # Forward pass
            if isinstance(model, StudentFECNet):
                # Student returns (embedding, attention_map)
                embeddings, _ = model(images, binary_mask=None)
            else:
                # Teacher returns embedding only
                embeddings = model(images)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_emotions.extend(emotions)
            all_emotion_indices.append(emotion_indices)
    
    embeddings = np.vstack(all_embeddings)
    emotion_indices = np.concatenate(all_emotion_indices)
    
    return embeddings, all_emotions, emotion_indices


def compute_umap(embeddings, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42):
    """
    Compute UMAP projection
    
    Args:
        embeddings: [N, D] array
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        metric: Distance metric
        random_state: Random seed
    
    Returns:
        umap_coords: [N, 2] UMAP coordinates
    """
    try:
        import umap
    except ImportError:
        raise ImportError("UMAP not installed. Install with: pip install umap-learn")
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric=metric,
        random_state=random_state,
        verbose=True
    )
    
    umap_coords = reducer.fit_transform(embeddings)
    return umap_coords


def plot_umap_comparison(teacher_coords, student_coords, emotions, emotion_indices, output_dir):
    """
    Create comparison plot of teacher vs student UMAP projections
    
    Args:
        teacher_coords: [N, 2] teacher UMAP coordinates
        student_coords: [N, 2] student UMAP coordinates
        emotions: [N] list of emotion labels
        emotion_indices: [N] emotion indices
        output_dir: Output directory path
    """
    # Emotion colors (distinct palette)
    emotion_colors = {
        'anger': '#e74c3c',      # Red
        'contempt': '#9b59b6',   # Purple
        'disgust': '#2ecc71',    # Green
        'fear': '#f39c12',       # Orange
        'happy': '#f1c40f',      # Yellow
        'neutral': '#95a5a6',    # Gray
        'sad': '#3498db',        # Blue
        'surprise': '#1abc9c'    # Teal
    }
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    
    # Plot teacher embeddings
    ax = axes[0]
    for emotion in emotion_colors.keys():
        mask = np.array([e == emotion for e in emotions])
        if mask.sum() > 0:
            ax.scatter(
                teacher_coords[mask, 0],
                teacher_coords[mask, 1],
                c=emotion_colors[emotion],
                label=emotion.capitalize(),
                alpha=0.6,
                s=30,
                edgecolors='none'
            )
    
    ax.set_title('Teacher Model (FECNet) - Embedding Space', fontsize=16, fontweight='bold')
    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Plot student embeddings
    ax = axes[1]
    for emotion in emotion_colors.keys():
        mask = np.array([e == emotion for e in emotions])
        if mask.sum() > 0:
            ax.scatter(
                student_coords[mask, 0],
                student_coords[mask, 1],
                c=emotion_colors[emotion],
                label=emotion.capitalize(),
                alpha=0.6,
                s=30,
                edgecolors='none'
            )
    
    ax.set_title('Student Model (OccFECNet) - Embedding Space', fontsize=16, fontweight='bold')
    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir) / 'umap_teacher_vs_student.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison plot: {output_path}")
    plt.close()
    
    # Create individual high-res plots
    for model_name, coords in [('teacher', teacher_coords), ('student', student_coords)]:
        fig, ax = plt.subplots(figsize=(12, 10))
        
        for emotion in emotion_colors.keys():
            mask = np.array([e == emotion for e in emotions])
            if mask.sum() > 0:
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    c=emotion_colors[emotion],
                    label=f'{emotion.capitalize()} (n={mask.sum()})',
                    alpha=0.6,
                    s=50,
                    edgecolors='white',
                    linewidths=0.5
                )
        
        title = 'Teacher Model (FECNet)' if model_name == 'teacher' else 'Student Model (OccFECNet)'
        ax.set_title(f'{title} - AffectNet Embedding Space', fontsize=18, fontweight='bold')
        ax.set_xlabel('UMAP Dimension 1', fontsize=14)
        ax.set_ylabel('UMAP Dimension 2', fontsize=14)
        ax.legend(loc='best', fontsize=11, framealpha=0.95, edgecolor='black')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#f8f9fa')
        
        output_path = Path(output_dir) / f'umap_{model_name}_detailed.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved {model_name} plot: {output_path}")
        plt.close()


def compute_embedding_statistics(teacher_emb, student_emb, emotions):
    """
    Compute statistics comparing teacher and student embeddings
    
    Args:
        teacher_emb: [N, 16] teacher embeddings
        student_emb: [N, 16] student embeddings
        emotions: [N] list of emotion labels
    
    Returns:
        stats: Dictionary of statistics
    """
    # Cosine similarity between teacher and student
    cos_sim = (teacher_emb * student_emb).sum(axis=1) / (
        np.linalg.norm(teacher_emb, axis=1) * np.linalg.norm(student_emb, axis=1) + 1e-8
    )
    
    # Per-emotion statistics
    emotion_stats = {}
    for emotion in np.unique(emotions):
        mask = np.array([e == emotion for e in emotions])
        emotion_stats[emotion] = {
            'count': mask.sum(),
            'mean_cosine_sim': cos_sim[mask].mean(),
            'std_cosine_sim': cos_sim[mask].std()
        }
    
    stats = {
        'overall_mean_cosine_sim': cos_sim.mean(),
        'overall_std_cosine_sim': cos_sim.std(),
        'per_emotion': emotion_stats
    }
    
    return stats


def save_statistics(stats, output_dir):
    """Save statistics to text file"""
    output_path = Path(output_dir) / 'embedding_statistics.txt'
    
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Teacher vs Student Embedding Statistics (AffectNet)\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Overall Cosine Similarity:\n")
        f.write(f"  Mean: {stats['overall_mean_cosine_sim']:.4f}\n")
        f.write(f"  Std:  {stats['overall_std_cosine_sim']:.4f}\n\n")
        
        f.write("Per-Emotion Statistics:\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Emotion':<12} {'Count':<8} {'Mean Cos Sim':<15} {'Std Cos Sim':<12}\n")
        f.write("-"*70 + "\n")
        
        for emotion, emotion_stat in stats['per_emotion'].items():
            f.write(f"{emotion.capitalize():<12} {emotion_stat['count']:<8} "
                   f"{emotion_stat['mean_cosine_sim']:<15.4f} "
                   f"{emotion_stat['std_cosine_sim']:<12.4f}\n")
    
    print(f"\nSaved statistics: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='UMAP Visualization of Teacher vs Student Embeddings')
    
    # Model paths
    parser.add_argument('--teacher_path', type=str, default='pretrained/FECNet.pt',
                       help='Path to teacher model')
    parser.add_argument('--student_path', type=str, default='checkpoints/curriculum/student_best.pth',
                       help='Path to student model checkpoint')
    
    # Data
    parser.add_argument('--csv_path', type=str, default='data/AffectNet/labels.csv',
                       help='Path to AffectNet labels.csv')
    parser.add_argument('--data_root', type=str, default='data/AffectNet',
                       help='AffectNet data root directory')
    parser.add_argument('--max_samples', type=int, default=5000,
                       help='Maximum samples to use (None = all, default: 5000)')
    
    # UMAP parameters
    parser.add_argument('--n_neighbors', type=int, default=15,
                       help='UMAP n_neighbors parameter')
    parser.add_argument('--min_dist', type=float, default=0.1,
                       help='UMAP min_dist parameter')
    parser.add_argument('--metric', type=str, default='cosine',
                       choices=['cosine', 'euclidean', 'manhattan'],
                       help='UMAP distance metric')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='umap',
                       help='Output directory')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for embedding extraction')
    
    # Device
    parser.add_argument('--device', type=int, default=0,
                       help='CUDA device')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("="*70)
    print("Loading AffectNet Dataset")
    print("="*70)
    dataset = AffectNetDataset(
        csv_path=args.csv_path,
        data_root=args.data_root,
        max_samples=args.max_samples
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Load teacher model
    print("\n" + "="*70)
    print("Loading Teacher Model")
    print("="*70)
    teacher = FECNet(pretrained=False)
    teacher = teacher.to(device)
    
    # Load weights manually
    checkpoint = torch.load(args.teacher_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        teacher.load_state_dict(checkpoint['model_state_dict'])
    else:
        teacher.load_state_dict(checkpoint)
    
    teacher.eval()
    print("Teacher model loaded")
    
    # Load student model
    print("\n" + "="*70)
    print("Loading Student Model")
    print("="*70)
    student = StudentFECNet(pretrained_teacher_path=args.teacher_path)
    checkpoint = torch.load(args.student_path, map_location=device)
    
    # Handle both checkpoint formats:
    # - student_epoch_*.pth has 'model_state_dict' key
    # - student_best.pth / student_final.pth are raw state dicts
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        student.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"Loaded checkpoint from epoch {epoch}")
    else:
        student.load_state_dict(checkpoint)
        print("Loaded checkpoint (raw state dict)")
    
    student = student.to(device)
    student.eval()
    print("Student model ready")
    
    # Extract embeddings
    print("\n" + "="*70)
    print("Extracting Embeddings")
    print("="*70)
    
    teacher_emb, emotions, emotion_indices = extract_embeddings(
        teacher, dataloader, device, "Teacher"
    )
    
    student_emb, _, _ = extract_embeddings(
        student, dataloader, device, "Student"
    )
    
    print(f"\nExtracted embeddings: {teacher_emb.shape}")
    
    # Compute statistics
    print("\n" + "="*70)
    print("Computing Statistics")
    print("="*70)
    stats = compute_embedding_statistics(teacher_emb, student_emb, emotions)
    save_statistics(stats, args.output_dir)
    
    print(f"\nOverall Mean Cosine Similarity: {stats['overall_mean_cosine_sim']:.4f}")
    
    # Compute UMAP projections
    print("\n" + "="*70)
    print("Computing UMAP Projection - Teacher")
    print("="*70)
    teacher_umap = compute_umap(
        teacher_emb,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric
    )
    
    print("\n" + "="*70)
    print("Computing UMAP Projection - Student")
    print("="*70)
    student_umap = compute_umap(
        student_emb,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric
    )
    
    # Create visualizations
    print("\n" + "="*70)
    print("Creating Visualizations")
    print("="*70)
    plot_umap_comparison(
        teacher_umap,
        student_umap,
        emotions,
        emotion_indices,
        args.output_dir
    )
    
    # Save UMAP coordinates
    np.savez(
        Path(args.output_dir) / 'umap_coordinates.npz',
        teacher_umap=teacher_umap,
        student_umap=student_umap,
        teacher_embeddings=teacher_emb,
        student_embeddings=student_emb,
        emotions=emotions,
        emotion_indices=emotion_indices
    )
    print(f"\nSaved UMAP coordinates: {args.output_dir}/umap_coordinates.npz")
    
    print("\n" + "="*70)
    print("UMAP Visualization Complete!")
    print("="*70)
    print(f"Output directory: {args.output_dir}/")
    print("Generated files:")
    print("  - umap_teacher_vs_student.png (comparison)")
    print("  - umap_teacher_detailed.png")
    print("  - umap_student_detailed.png")
    print("  - embedding_statistics.txt")
    print("  - umap_coordinates.npz")


if __name__ == '__main__':
    main()

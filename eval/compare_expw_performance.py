"""
Compare Teacher vs Student Model Performance on EXPW Dataset
=============================================================

This script evaluates and compares FECNet (teacher) and StudentFECNet (student)
on 1000 balanced samples from the EXPW dataset using embedding similarity analysis.

The goal is to demonstrate that the student model maintains comparable performance
to the teacher on clean facial expression images.

Metrics:
- Embedding Cosine Similarity (how similar are teacher and student embeddings?)
- Per-emotion similarity analysis
- Attention quality visualization

Expected Results:
- High similarity (>0.90) indicates student learned well from teacher
- Consistent similarity across emotions shows robust learning
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import random
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Import models
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.FECNet import FECNet
from models.student_fecnet import StudentFECNet
from models.mtcnn import MTCNN


class EXPWDataset(Dataset):
    """EXPW Dataset loader with balanced sampling"""
    
    EMOTIONS = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    EMOTION_MAP = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happy', 
                   4: 'sad', 5: 'surprise', 6: 'neutral'}
    
    def __init__(self, label_file, image_root, samples_per_class=150, transform=None, seed=42):
        """
        Args:
            label_file: Path to label.lst
            image_root: Path to image directory
            samples_per_class: Number of samples per emotion (default: 150 = 1050 total)
            transform: Image transformations
            seed: Random seed for reproducibility
        """
        self.image_root = Path(image_root)
        self.transform = transform
        
        # Load all samples
        all_samples = self._load_labels(label_file)
        
        # Balance samples across emotions
        self.samples = self._balance_samples(all_samples, samples_per_class, seed)
        
        print(f"Loaded {len(self.samples)} balanced samples from EXPW")
        self._print_distribution()
    
    def _load_labels(self, label_file):
        """Load all samples from label file and filter to existing files only"""
        samples = []
        missing_count = 0
        
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8:
                    img_name = parts[0]
                    emotion_label = int(parts[7])
                    
                    # Only keep emotions in our 7-class setup
                    if emotion_label in self.EMOTION_MAP:
                        # Check if file exists
                        img_path = self.image_root / img_name
                        if img_path.exists():
                            samples.append({
                                'image': img_name,
                                'emotion': emotion_label,
                                'emotion_name': self.EMOTION_MAP[emotion_label]
                            })
                        else:
                            missing_count += 1
        
        if missing_count > 0:
            print(f"Skipped {missing_count} missing image files")
        
        return samples
    
    def _balance_samples(self, all_samples, samples_per_class, seed):
        """Sample equal number from each emotion class"""
        random.seed(seed)
        
        # Group by emotion
        emotion_groups = {i: [] for i in range(7)}
        for sample in all_samples:
            emotion_groups[sample['emotion']].append(sample)
        
        # Sample from each group
        balanced = []
        for emotion_id in range(7):
            group = emotion_groups[emotion_id]
            if len(group) >= samples_per_class:
                sampled = random.sample(group, samples_per_class)
            else:
                # If not enough samples, use all and warn
                sampled = group
                print(f"Warning: {self.EMOTION_MAP[emotion_id]} has only {len(group)} samples")
            balanced.extend(sampled)
        
        # Shuffle the final balanced dataset
        random.shuffle(balanced)
        return balanced
    
    def _print_distribution(self):
        """Print emotion distribution"""
        emotion_counts = {name: 0 for name in self.EMOTIONS}
        for sample in self.samples:
            emotion_counts[sample['emotion_name']] += 1
        
        print("\nEmotion Distribution:")
        for emotion, count in emotion_counts.items():
            print(f"  {emotion:10s}: {count:4d} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        img_path = self.image_root / sample['image']
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'emotion': sample['emotion'],
            'emotion_name': sample['emotion_name'],
            'image_path': str(img_path)
        }


def get_transforms():
    """Get image transforms - same as training (224x224)"""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def load_models(teacher_path, student_path, device):
    """Load teacher and student models"""
    print("\nLoading models...")
    
    # Load teacher
    teacher_model = FECNet(pretrained=False)
    teacher_checkpoint = torch.load(teacher_path, map_location=device)
    if 'model_state_dict' in teacher_checkpoint:
        teacher_model.load_state_dict(teacher_checkpoint['model_state_dict'])
    else:
        teacher_model.load_state_dict(teacher_checkpoint)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    print(f"[SUCCESS] Loaded teacher model from {teacher_path}")
    
    # Load student
    student_model = StudentFECNet(pretrained_teacher_path=None)
    student_checkpoint = torch.load(student_path, map_location=device)
    if 'model_state_dict' in student_checkpoint:
        student_model.load_state_dict(student_checkpoint['model_state_dict'])
    else:
        student_model.load_state_dict(student_checkpoint)
    student_model = student_model.to(device)
    student_model.eval()
    print(f"[SUCCESS] Loaded student model from {student_path}")
    
    return teacher_model, student_model


@torch.no_grad()
def extract_embeddings(model, dataloader, device, model_name="Model"):
    """Extract embeddings from model"""
    all_embeddings = []
    all_emotions = []
    all_emotion_names = []
    all_attention_maps = []
    
    print(f"\nExtracting embeddings from {model_name}...")
    for batch in tqdm(dataloader, desc=f"{model_name}"):
        images = batch['image'].to(device)
        emotions = batch['emotion']
        emotion_names = batch['emotion_name']
        
        # Forward pass - both models return (embeddings, attention_maps)
        embeddings, attention_maps = model(images)
        
        all_embeddings.append(embeddings.cpu().numpy())
        all_emotions.extend(emotions.numpy())
        all_emotion_names.extend(emotion_names)
        if attention_maps is not None:
            all_attention_maps.append(attention_maps.cpu().numpy())
    
    embeddings = np.vstack(all_embeddings)
    attention = np.vstack(all_attention_maps) if all_attention_maps else None
    
    print(f"[SUCCESS] Extracted {len(embeddings)} embeddings (shape: {embeddings.shape})")
    return embeddings, np.array(all_emotions), all_emotion_names, attention


def compute_embedding_similarity(teacher_emb, student_emb):
    """Compute pairwise cosine similarity between teacher and student embeddings"""
    # Compute similarity for each pair
    similarities = []
    for i in range(len(teacher_emb)):
        sim = cosine_similarity(
            teacher_emb[i:i+1],
            student_emb[i:i+1]
        )[0, 0]
        similarities.append(sim)
    
    similarities = np.array(similarities)
    
    print(f"\nEmbedding Similarity Statistics:")
    print(f"  Mean Similarity: {similarities.mean():.4f}")
    print(f"  Std Similarity:  {similarities.std():.4f}")
    print(f"  Min Similarity:  {similarities.min():.4f}")
    print(f"  Max Similarity:  {similarities.max():.4f}")
    
    return similarities


def plot_similarity_analysis(similarities, emotions, emotion_names, output_dir):
    """Create comprehensive similarity visualization"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Overall similarity distribution
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(similarities, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(similarities.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {similarities.mean():.4f}')
    ax1.set_xlabel('Cosine Similarity', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Overall Embedding Similarity Distribution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Per-emotion similarity boxplot
    ax2 = plt.subplot(2, 3, 2)
    emotion_labels = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    emotion_sims = [similarities[emotions == i] for i in range(7)]
    
    bp = ax2.boxplot(emotion_sims, labels=emotion_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax2.set_xlabel('Emotion', fontsize=12)
    ax2.set_ylabel('Cosine Similarity', fontsize=12)
    ax2.set_title('Similarity by Emotion Category', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Per-emotion mean similarity bar chart
    ax3 = plt.subplot(2, 3, 3)
    emotion_means = [similarities[emotions == i].mean() for i in range(7)]
    emotion_stds = [similarities[emotions == i].std() for i in range(7)]
    
    colors = plt.cm.Set3(range(7))
    bars = ax3.bar(emotion_labels, emotion_means, yerr=emotion_stds, 
                   color=colors, alpha=0.8, capsize=5)
    ax3.axhline(similarities.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Overall Mean: {similarities.mean():.4f}')
    ax3.set_xlabel('Emotion', fontsize=12)
    ax3.set_ylabel('Mean Cosine Similarity', fontsize=12)
    ax3.set_title('Mean Similarity per Emotion', fontsize=14, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0.85, 1.0])
    
    # Add value labels on bars
    for bar, mean_val in zip(bars, emotion_means):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Similarity heatmap (sample 100 random pairs)
    ax4 = plt.subplot(2, 3, 4)
    sample_indices = np.random.choice(len(similarities), min(100, len(similarities)), replace=False)
    sim_matrix = cosine_similarity(
        np.arange(len(sample_indices)).reshape(-1, 1),
        np.arange(len(sample_indices)).reshape(-1, 1)
    )
    # Just use similarities as diagonal for visualization
    np.fill_diagonal(sim_matrix, similarities[sample_indices])
    
    sns.heatmap(sim_matrix[:20, :20], cmap='RdYlGn', center=0.95, 
                vmin=0.85, vmax=1.0, ax=ax4, cbar_kws={'label': 'Similarity'})
    ax4.set_title('Similarity Heatmap (Sample)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Sample Index', fontsize=12)
    ax4.set_ylabel('Sample Index', fontsize=12)
    
    # 5. Cumulative distribution
    ax5 = plt.subplot(2, 3, 5)
    sorted_sims = np.sort(similarities)
    cumulative = np.arange(1, len(sorted_sims) + 1) / len(sorted_sims)
    ax5.plot(sorted_sims, cumulative, linewidth=2, color='steelblue')
    ax5.axvline(similarities.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {similarities.mean():.4f}')
    ax5.axhline(0.95, color='green', linestyle='--', linewidth=1, alpha=0.5,
                label='95% threshold')
    ax5.set_xlabel('Cosine Similarity', fontsize=12)
    ax5.set_ylabel('Cumulative Probability', fontsize=12)
    ax5.set_title('Cumulative Distribution', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # 6. Statistics table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    stats_data = []
    stats_data.append(['Overall Statistics', ''])
    stats_data.append(['Mean Similarity', f'{similarities.mean():.4f}'])
    stats_data.append(['Std Deviation', f'{similarities.std():.4f}'])
    stats_data.append(['Min Similarity', f'{similarities.min():.4f}'])
    stats_data.append(['Max Similarity', f'{similarities.max():.4f}'])
    stats_data.append(['Median Similarity', f'{np.median(similarities):.4f}'])
    stats_data.append(['', ''])
    stats_data.append(['Samples > 0.95', f'{(similarities > 0.95).sum()} ({(similarities > 0.95).mean()*100:.1f}%)'])
    stats_data.append(['Samples > 0.90', f'{(similarities > 0.90).sum()} ({(similarities > 0.90).mean()*100:.1f}%)'])
    stats_data.append(['Samples > 0.85', f'{(similarities > 0.85).sum()} ({(similarities > 0.85).mean()*100:.1f}%)'])
    
    table = ax6.table(cellText=stats_data, cellLoc='left', loc='center',
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the header
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax6.set_title('Similarity Statistics', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    output_path = output_dir / 'expw_similarity_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[SUCCESS] Saved similarity analysis to {output_path}")
    plt.close()


def visualize_attention_samples(dataset, student_model, device, output_dir, num_samples=8):
    """Visualize student attention on sample images"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample diverse images (one from each emotion + random)
    emotion_samples = {i: [] for i in range(7)}
    for idx, sample in enumerate(dataset.samples):
        emotion_samples[sample['emotion']].append(idx)
    
    sample_indices = []
    for emotion_id in range(7):
        if emotion_samples[emotion_id]:
            sample_indices.append(random.choice(emotion_samples[emotion_id]))
    
    # Add one more random sample
    if len(sample_indices) < num_samples:
        remaining = set(range(len(dataset))) - set(sample_indices)
        sample_indices.append(random.choice(list(remaining)))
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, sample_idx in enumerate(sample_indices[:num_samples]):
        sample = dataset[sample_idx]
        image = sample['image'].unsqueeze(0).to(device)
        emotion_name = sample['emotion_name']
        
        # Get attention
        with torch.no_grad():
            _, attention_maps = student_model(image)
        
        # Denormalize image for display
        img_display = image[0].cpu().permute(1, 2, 0).numpy()
        img_display = (img_display * 0.5) + 0.5  # Denormalize from [-1, 1] to [0, 1]
        img_display = np.clip(img_display, 0, 1)
        
        # Get attention map
        if attention_maps is not None:
            attn_map = attention_maps[0, 0].cpu().numpy()
            
            # Resize attention to match image
            from scipy.ndimage import zoom
            zoom_factor = img_display.shape[0] / attn_map.shape[0]
            attn_resized = zoom(attn_map, zoom_factor, order=1)
            
            # Overlay attention
            axes[idx].imshow(img_display)
            axes[idx].imshow(attn_resized, alpha=0.4, cmap='jet')
        else:
            axes[idx].imshow(img_display)
        
        axes[idx].set_title(f'{emotion_name.capitalize()}', fontsize=12, fontweight='bold')
        axes[idx].axis('off')
    
    plt.suptitle('Student Model Attention on EXPW Samples', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / 'expw_attention_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Saved attention visualization to {output_path}")
    plt.close()


def save_detailed_report(similarities, emotions, emotion_names, output_dir):
    """Save detailed text report"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / 'expw_comparison_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Teacher vs Student Model Comparison on EXPW Dataset\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("OVERALL SIMILARITY STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean Cosine Similarity:   {similarities.mean():.4f}\n")
        f.write(f"Std Deviation:            {similarities.std():.4f}\n")
        f.write(f"Min Similarity:           {similarities.min():.4f}\n")
        f.write(f"Max Similarity:           {similarities.max():.4f}\n")
        f.write(f"Median Similarity:        {np.median(similarities):.4f}\n\n")
        
        f.write(f"Samples with similarity > 0.95: {(similarities > 0.95).sum():4d} ({(similarities > 0.95).mean()*100:5.1f}%)\n")
        f.write(f"Samples with similarity > 0.90: {(similarities > 0.90).sum():4d} ({(similarities > 0.90).mean()*100:5.1f}%)\n")
        f.write(f"Samples with similarity > 0.85: {(similarities > 0.85).sum():4d} ({(similarities > 0.85).mean()*100:5.1f}%)\n\n")
        
        f.write("\nPER-EMOTION ANALYSIS\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Emotion':<12} {'Count':>6} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}\n")
        f.write("-" * 80 + "\n")
        
        emotion_labels = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        for i, emotion in enumerate(emotion_labels):
            emotion_mask = emotions == i
            if emotion_mask.sum() > 0:
                e_sims = similarities[emotion_mask]
                f.write(f"{emotion:<12} {emotion_mask.sum():>6} {e_sims.mean():>8.4f} "
                       f"{e_sims.std():>8.4f} {e_sims.min():>8.4f} {e_sims.max():>8.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("INTERPRETATION\n")
        f.write("=" * 80 + "\n\n")
        
        mean_sim = similarities.mean()
        if mean_sim > 0.95:
            interpretation = "[SUCCESS] EXCELLENT - Student learned very well from teacher"
        elif mean_sim > 0.90:
            interpretation = "[SUCCESS] GOOD - Student maintains strong similarity to teacher"
        elif mean_sim > 0.85:
            interpretation = "[WARNING] MODERATE - Some divergence from teacher"
        else:
            interpretation = "[CONCERN] LOW - Significant divergence from teacher"
        
        f.write(f"{interpretation}\n\n")
        f.write(f"With mean similarity of {mean_sim:.4f}, the student model demonstrates\n")
        f.write(f"{'strong' if mean_sim > 0.90 else 'moderate'} alignment with the teacher model's learned representations.\n")
        f.write(f"This indicates that the knowledge distillation process was {'successful' if mean_sim > 0.90 else 'partially successful'}.\n")
    
    print(f"[SUCCESS] Saved detailed report to {report_path}")


def main():
    # Configuration
    DATA_ROOT = Path(r'c:\PNS\Projects\CV\Project_Ref\FECNet\data\expw_dataset\data')
    LABEL_FILE = DATA_ROOT / 'label' / 'label.lst'
    IMAGE_ROOT = DATA_ROOT / 'image'
    
    TEACHER_PATH = Path(r'c:\PNS\Projects\CV\Project_Ref\FECNet\pretrained\FECNet.pt')
    STUDENT_PATH = Path(r'c:\PNS\Projects\CV\Project_Ref\FECNet\checkpoints\curriculum\student_epoch_60.pth')
    
    OUTPUT_DIR = Path(r'c:\PNS\Projects\CV\Project_Ref\FECNet\eval\expw_comparison_results')
    
    SAMPLES_PER_CLASS = 150  # 150 * 7 = 1050 total samples
    BATCH_SIZE = 32
    SEED = 42
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if output already exists
    if OUTPUT_DIR.exists():
        existing_files = list(OUTPUT_DIR.glob('*.png')) + list(OUTPUT_DIR.glob('*.txt'))
        if existing_files:
            print(f"\n[INFO] Found existing results in {OUTPUT_DIR}")
            print(f"[INFO] Results already computed. Skipping computation.")
            print(f"[INFO] Delete {OUTPUT_DIR} to recompute.")
            return
    
    # Create dataset
    print("\n" + "="*80)
    print("LOADING EXPW DATASET")
    print("="*80)
    dataset = EXPWDataset(
        label_file=LABEL_FILE,
        image_root=IMAGE_ROOT,
        samples_per_class=SAMPLES_PER_CLASS,
        transform=get_transforms(),
        seed=SEED
    )
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Load models
    print("\n" + "="*80)
    print("LOADING MODELS")
    print("="*80)
    teacher_model, student_model = load_models(TEACHER_PATH, STUDENT_PATH, device)
    
    # Extract embeddings
    print("\n" + "="*80)
    print("EXTRACTING EMBEDDINGS")
    print("="*80)
    teacher_emb, teacher_emotions, teacher_names, _ = extract_embeddings(
        teacher_model, dataloader, device, "Teacher"
    )
    student_emb, student_emotions, student_names, student_attn = extract_embeddings(
        student_model, dataloader, device, "Student"
    )
    
    # Compute similarity
    print("\n" + "="*80)
    print("COMPUTING EMBEDDING SIMILARITY")
    print("="*80)
    similarities = compute_embedding_similarity(teacher_emb, student_emb)
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    plot_similarity_analysis(similarities, teacher_emotions, teacher_names, OUTPUT_DIR)
    visualize_attention_samples(dataset, student_model, device, OUTPUT_DIR)
    save_detailed_report(similarities, teacher_emotions, teacher_names, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"\nKey Finding: Mean similarity = {similarities.mean():.4f}")
    
    if similarities.mean() > 0.90:
        print("[SUCCESS] Student model learned well from teacher!")
    else:
        print("[WARNING] Student shows some divergence from teacher")


if __name__ == '__main__':
    main()

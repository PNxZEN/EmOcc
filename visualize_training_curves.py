"""
Visualize Training Curves for Curriculum Learning
Generates comprehensive plots from training logs

Usage:
    python visualize_training_curves.py --log_dir logs/curriculum --experiment <name>
    python visualize_training_curves.py --log_dir logs/curriculum --compare exp1 exp2 exp3
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


def load_logs(log_dir, experiment_name):
    """
    Load training and evaluation logs
    
    Args:
        log_dir: Directory containing logs
        experiment_name: Name of experiment
    
    Returns:
        train_df, eval_df: DataFrames with training and evaluation data
    """
    log_dir = Path(log_dir)
    
    train_path = log_dir / f"{experiment_name}_train.csv"
    eval_path = log_dir / f"{experiment_name}_eval.csv"
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training log not found: {train_path}")
    
    train_df = pd.read_csv(train_path)
    
    eval_df = None
    if eval_path.exists():
        eval_df = pd.read_csv(eval_path)
    
    return train_df, eval_df


def plot_training_curves(train_df, eval_df, output_path, experiment_name):
    """
    Generate comprehensive training curves plot
    
    Creates 6 subplots:
    1. Total Loss
    2. Loss Components (distillation, consistency)
    3. Attention Losses (reg, div)
    4. Beta Parameter Evolution
    5. Lambda Values (curriculum schedule)
    6. Evaluation Cosine Similarity
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'Training Curves: {experiment_name}', fontsize=16, fontweight='bold')
    
    # Color palette for phases
    phase_colors = {1: '#3498db', 2: '#e74c3c', 3: '#f39c12', 4: '#2ecc71'}
    
    # Add phase background shading
    def add_phase_shading(ax, train_df):
        phases = train_df.groupby('phase')['epoch'].agg(['min', 'max'])
        for phase, (min_epoch, max_epoch) in phases.iterrows():
            ax.axvspan(min_epoch, max_epoch, alpha=0.1, color=phase_colors[phase])
            # Add phase label
            mid_epoch = (min_epoch + max_epoch) / 2
            ax.text(mid_epoch, ax.get_ylim()[1] * 0.95, f'Phase {phase}',
                   ha='center', va='top', fontsize=9, fontweight='bold',
                   color=phase_colors[phase], alpha=0.7)
    
    # 1. Total Loss
    ax1 = axes[0, 0]
    ax1.plot(train_df['epoch'], train_df['loss'], linewidth=2, color='darkblue', label='Total Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Total Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    add_phase_shading(ax1, train_df)
    
    # 2. Loss Components
    ax2 = axes[0, 1]
    ax2.plot(train_df['epoch'], train_df['distillation'], linewidth=2, label='Distillation (lambda1)', color='#e74c3c')
    ax2.plot(train_df['epoch'], train_df['consistency'], linewidth=2, label='Consistency (lambda2)', color='#3498db')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Embedding Loss Components')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    add_phase_shading(ax2, train_df)
    
    # 3. Attention Losses
    ax3 = axes[1, 0]
    ax3.plot(train_df['epoch'], train_df['attention_reg'], linewidth=2, label='Attention Reg (lambda3)', color='#f39c12')
    ax3.plot(train_df['epoch'], train_df['attention_div'], linewidth=2, label='Attention Div (lambda4)', color='#9b59b6')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('Attention Loss Components')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    add_phase_shading(ax3, train_df)
    
    # 4. Beta Parameter
    ax4 = axes[1, 1]
    ax4.plot(train_df['epoch'], train_df['beta'], linewidth=2, color='#2ecc71', marker='o', markersize=3)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Beta Value')
    ax4.set_title('Residual Connection Strength (beta)')
    ax4.grid(True, alpha=0.3)
    add_phase_shading(ax4, train_df)
    # Add horizontal line at 0
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 5. Lambda Values (Curriculum Schedule)
    ax5 = axes[2, 0]
    ax5.plot(train_df['epoch'], train_df['lambda1'], linewidth=2, label='lambda1 (Distillation)', color='#e74c3c')
    ax5.plot(train_df['epoch'], train_df['lambda2'], linewidth=2, label='lambda2 (Consistency)', color='#3498db')
    ax5.plot(train_df['epoch'], train_df['lambda3'], linewidth=2, label='lambda3 (Attn Reg)', color='#f39c12')
    ax5.plot(train_df['epoch'], train_df['lambda4'], linewidth=2, label='lambda4 (Attn Div)', color='#9b59b6')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Lambda Value')
    ax5.set_title('Loss Weight Curriculum Schedule')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    add_phase_shading(ax5, train_df)
    
    # 6. Evaluation Cosine Similarity
    ax6 = axes[2, 1]
    if eval_df is not None and len(eval_df) > 0:
        ax6.plot(eval_df['epoch'], eval_df['cosine_similarity'], linewidth=2, 
                marker='o', markersize=6, color='#2ecc71', label='Cosine Similarity')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Cosine Similarity')
        ax6.set_title('Validation: Teacher-Student Embedding Similarity')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim([0, 1])
        add_phase_shading(ax6, train_df)
    else:
        ax6.text(0.5, 0.5, 'No Evaluation Data', ha='center', va='center', fontsize=14)
        ax6.set_title('Validation Metrics')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[Visualization] Saved training curves: {output_path}")
    plt.close()


def plot_attention_metrics(train_df, output_path, experiment_name):
    """
    Plot attention-specific metrics (Phase 3-4 only)
    
    Creates 3 subplots:
    1. Attention-Mask Correlation
    2. Attention Sparsity
    3. Attention Entropy
    """
    # Filter for Phase 3-4 where attention metrics exist
    phase34_df = train_df[train_df['phase'] >= 3].copy()
    
    if len(phase34_df) == 0 or phase34_df['attn_correlation'].isna().all():
        print("[Visualization] No attention metrics to plot (Phase 1-2 only)")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Attention Metrics (Phase 3-4): {experiment_name}', fontsize=16, fontweight='bold')
    
    phase_colors = {3: '#f39c12', 4: '#2ecc71'}
    
    def add_phase_shading(ax, df):
        phases = df.groupby('phase')['epoch'].agg(['min', 'max'])
        for phase, (min_epoch, max_epoch) in phases.iterrows():
            ax.axvspan(min_epoch, max_epoch, alpha=0.1, color=phase_colors[phase])
            mid_epoch = (min_epoch + max_epoch) / 2
            ax.text(mid_epoch, ax.get_ylim()[1] * 0.95, f'Phase {phase}',
                   ha='center', va='top', fontsize=9, fontweight='bold',
                   color=phase_colors[phase], alpha=0.7)
    
    # 1. Correlation
    ax1 = axes[0]
    ax1.plot(phase34_df['epoch'], phase34_df['attn_correlation'], linewidth=2, color='#3498db')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Correlation')
    ax1.set_title('Attention-Mask Correlation')
    ax1.grid(True, alpha=0.3)
    add_phase_shading(ax1, phase34_df)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 2. Sparsity
    ax2 = axes[1]
    ax2.plot(phase34_df['epoch'], phase34_df['attn_sparsity'], linewidth=2, color='#e74c3c')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Sparsity')
    ax2.set_title('Attention Sparsity')
    ax2.grid(True, alpha=0.3)
    add_phase_shading(ax2, phase34_df)
    
    # 3. Entropy
    ax3 = axes[2]
    ax3.plot(phase34_df['epoch'], phase34_df['attn_entropy'], linewidth=2, color='#9b59b6')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Entropy')
    ax3.set_title('Attention Entropy')
    ax3.grid(True, alpha=0.3)
    add_phase_shading(ax3, phase34_df)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[Visualization] Saved attention metrics: {output_path}")
    plt.close()


def plot_comparison(experiments, log_dir, output_dir):
    """
    Compare multiple experiments on the same plot
    
    Args:
        experiments: List of experiment names
        log_dir: Directory containing logs
        output_dir: Directory to save comparison plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Experiment Comparison', fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    
    for idx, exp_name in enumerate(experiments):
        try:
            train_df, eval_df = load_logs(log_dir, exp_name)
            color = colors[idx]
            
            # Total Loss
            axes[0, 0].plot(train_df['epoch'], train_df['loss'], 
                           linewidth=2, color=color, label=exp_name, alpha=0.8)
            
            # Distillation Loss
            axes[0, 1].plot(train_df['epoch'], train_df['distillation'],
                           linewidth=2, color=color, label=exp_name, alpha=0.8)
            
            # Beta
            axes[1, 0].plot(train_df['epoch'], train_df['beta'],
                           linewidth=2, color=color, label=exp_name, alpha=0.8)
            
            # Cosine Similarity
            if eval_df is not None and len(eval_df) > 0:
                axes[1, 1].plot(eval_df['epoch'], eval_df['cosine_similarity'],
                               linewidth=2, marker='o', markersize=4, color=color, 
                               label=exp_name, alpha=0.8)
        
        except FileNotFoundError as e:
            print(f"[Warning] Could not load {exp_name}: {e}")
    
    # Configure subplots
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Distillation Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Beta Value')
    axes[1, 0].set_title('Residual Connection Strength (beta)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Cosine Similarity')
    axes[1, 1].set_title('Validation: Embedding Similarity')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    output_path = output_dir / 'comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[Visualization] Saved comparison: {output_path}")
    plt.close()


def generate_summary_stats(train_df, eval_df, experiment_name):
    """
    Generate summary statistics
    
    Returns:
        Dictionary with key statistics
    """
    stats = {
        'experiment': experiment_name,
        'total_epochs': len(train_df),
        'final_loss': train_df['loss'].iloc[-1],
        'final_distillation': train_df['distillation'].iloc[-1],
        'final_consistency': train_df['consistency'].iloc[-1],
        'final_beta': train_df['beta'].iloc[-1],
        'min_loss': train_df['loss'].min(),
        'min_loss_epoch': train_df['loss'].idxmin() + 1
    }
    
    if eval_df is not None and len(eval_df) > 0:
        stats['best_cosine_sim'] = eval_df['cosine_similarity'].max()
        stats['best_cosine_epoch'] = eval_df.loc[eval_df['cosine_similarity'].idxmax(), 'epoch']
        stats['final_cosine_sim'] = eval_df['cosine_similarity'].iloc[-1]
    
    # Phase-specific stats
    for phase in range(1, 5):
        phase_df = train_df[train_df['phase'] == phase]
        if len(phase_df) > 0:
            stats[f'phase{phase}_avg_loss'] = phase_df['loss'].mean()
            stats[f'phase{phase}_epochs'] = len(phase_df)
    
    return stats


def print_summary(stats):
    """Print summary statistics"""
    print("\n" + "="*70)
    print(f"TRAINING SUMMARY: {stats['experiment']}")
    print("="*70)
    print(f"Total Epochs: {stats['total_epochs']}")
    print(f"\nFinal Metrics:")
    print(f"  Total Loss:      {stats['final_loss']:.6f}")
    print(f"  Distillation:    {stats['final_distillation']:.6f}")
    print(f"  Consistency:     {stats['final_consistency']:.6f}")
    print(f"  Beta:            {stats['final_beta']:.6f}")
    
    if 'best_cosine_sim' in stats:
        print(f"\nValidation:")
        print(f"  Best Cosine Sim: {stats['best_cosine_sim']:.6f} (epoch {stats['best_cosine_epoch']:.0f})")
        print(f"  Final Cosine Sim: {stats['final_cosine_sim']:.6f}")
    
    print(f"\nBest Training:")
    print(f"  Min Loss:        {stats['min_loss']:.6f} (epoch {stats['min_loss_epoch']})")
    
    print(f"\nPhase Statistics:")
    for phase in range(1, 5):
        if f'phase{phase}_avg_loss' in stats:
            print(f"  Phase {phase}: {stats[f'phase{phase}_epochs']} epochs, "
                  f"avg loss {stats[f'phase{phase}_avg_loss']:.6f}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize Training Curves')
    parser.add_argument('--log_dir', type=str, default='logs/curriculum',
                       help='Directory containing training logs')
    parser.add_argument('--experiment', type=str,
                       help='Experiment name to visualize')
    parser.add_argument('--compare', nargs='+',
                       help='List of experiment names to compare')
    parser.add_argument('--output_dir', type=str, default='figures/training',
                       help='Directory to save plots')
    parser.add_argument('--list', action='store_true',
                       help='List available experiments')
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # List available experiments
    if args.list:
        print("\nAvailable experiments:")
        train_logs = list(log_dir.glob('*_train.csv'))
        for log_file in train_logs:
            exp_name = log_file.stem.replace('_train', '')
            print(f"  - {exp_name}")
        return
    
    # Single experiment visualization
    if args.experiment:
        print(f"\nVisualizing experiment: {args.experiment}")
        
        train_df, eval_df = load_logs(log_dir, args.experiment)
        
        # Generate plots
        curves_path = output_dir / f'{args.experiment}_curves.png'
        plot_training_curves(train_df, eval_df, curves_path, args.experiment)
        
        attention_path = output_dir / f'{args.experiment}_attention.png'
        plot_attention_metrics(train_df, attention_path, args.experiment)
        
        # Generate and print summary
        stats = generate_summary_stats(train_df, eval_df, args.experiment)
        print_summary(stats)
        
        print(f"\nPlots saved to: {output_dir}")
    
    # Comparison mode
    elif args.compare:
        print(f"\nComparing {len(args.compare)} experiments:")
        for exp in args.compare:
            print(f"  - {exp}")
        
        plot_comparison(args.compare, log_dir, output_dir)
        
        # Print summary for each
        for exp_name in args.compare:
            try:
                train_df, eval_df = load_logs(log_dir, exp_name)
                stats = generate_summary_stats(train_df, eval_df, exp_name)
                print_summary(stats)
            except FileNotFoundError as e:
                print(f"[Warning] Could not load {exp_name}: {e}")
    
    else:
        print("Error: Specify --experiment <name> or --compare <name1> <name2> ...")
        print("       Use --list to see available experiments")


if __name__ == '__main__':
    main()

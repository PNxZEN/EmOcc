"""
4-Phase Progressive Curriculum Training for Student FECNet
Reference: OccFECNet.md - Section 4.2
"""

import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.FECNet import FECNet
from models.student_fecnet import StudentFECNet
from utils.distillation_losses import MultiComponentLoss, compute_attention_metrics
from datasets.paired_occlusion_dataset import PairedOcclusionDataset
from utils.training_logger import TrainingLogger


def get_phase_config(phase, epoch):
    """
    Get configuration for each training phase
    Reference: OccFECNet.md Section 4.2
    
    Phase 1 (epochs 1-10): Baseline, clean only, beta frozen
    Phase 2 (epochs 11-20): Introduce occlusion, beta frozen  
    Phase 3 (epochs 21-40): Activate attention, progressive lambda3
    Phase 4 (epochs 41-60): Full training
    
    Args:
        phase: Phase number (1-4)
        epoch: Current epoch (1-60)
    
    Returns:
        dict with phase configuration
    """
    if phase == 1:
        # Phase 1: Baseline (epochs 1-10)
        return {
            'phase': 1,
            'epochs': (1, 10),
            'attention_frozen': True,
            'lambda1': 1.0,
            'lambda2': 0.0,  # No consistency loss yet
            'lambda3': 0.0,
            'lambda4': 0.0,
            'lr_densenet': 5e-4,
            'lr_attention': 0.0,  # Frozen
            'lr_beta': 0.0,  # Frozen
            'use_occluded': False,  # Clean only
            'description': 'Baseline distillation on clean faces'
        }
    
    elif phase == 2:
        # Phase 2: Introduce occlusion (epochs 11-20)
        return {
            'phase': 2,
            'epochs': (11, 20),
            'attention_frozen': True,
            'lambda1': 1.0,
            'lambda2': 0.5,
            'lambda3': 0.0,
            'lambda4': 0.0,
            'lr_densenet': 5e-4,
            'lr_attention': 0.0,  # Still frozen
            'lr_beta': 0.0,  # Still frozen
            'use_occluded': True,
            'description': 'Introduce occlusion with frozen attention'
        }
    
    elif phase == 3:
        # Phase 3: Activate attention (epochs 21-40)
        # Progressive lambda3: 0.0 -> 0.05 (epochs 21-30) -> 0.1 (epochs 31-40)
        if epoch <= 30:
            lambda3 = 0.05 * (epoch - 20) / 10  # Linear 0.0 -> 0.05
        else:
            lambda3 = 0.05 + 0.05 * (epoch - 30) / 10  # Linear 0.05 -> 0.1
        
        return {
            'phase': 3,
            'epochs': (21, 40),
            'attention_frozen': False,
            'lambda1': 1.0,
            'lambda2': 0.5,
            'lambda3': lambda3,
            'lambda4': 0.01,
            'lr_densenet': 5e-4,
            'lr_attention': 1e-4,
            'lr_beta': 1e-5,
            'use_occluded': True,
            'description': f'Activate attention (lambda3={lambda3:.3f})'
        }
    
    else:  # phase == 4
        # Phase 4: Full training (epochs 41-100)
        return {
            'phase': 4,
            'epochs': (41, 100),
            'attention_frozen': False,
            'lambda1': 1.0,
            'lambda2': 0.5,
            'lambda3': 0.1,
            'lambda4': 0.01,
            'lr_densenet': 5e-4,
            'lr_attention': 1e-4,
            'lr_beta': 1e-5,
            'use_occluded': True,
            'description': 'Full training with all components'
        }


def determine_phase(epoch):
    """Determine which phase we're in based on epoch number"""
    if epoch <= 10:
        return 1
    elif epoch <= 20:
        return 2
    elif epoch <= 40:
        return 3
    else:
        return 4


def train_epoch(student, teacher, dataloader, optimizer, loss_fn, device, config, epoch):
    """
    Train for one epoch
    
    Args:
        student: Student model
        teacher: Teacher model (frozen)
        dataloader: Training dataloader
        optimizer: Optimizer
        loss_fn: MultiComponentLoss
        device: CUDA device
        config: Phase configuration
        epoch: Current epoch number
    
    Returns:
        dict with training metrics
    """
    student.train()
    teacher.eval()
    
    total_loss = 0.0
    total_distill = 0.0
    total_consistency = 0.0
    total_attn_reg = 0.0
    total_attn_div = 0.0
    num_batches = 0
    
    # Attention metrics (if phase 3 or 4)
    if config['phase'] >= 3:
        total_correlation = 0.0
        total_sparsity = 0.0
        total_entropy = 0.0
        num_attention_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Phase {config['phase']}]")
    
    for batch in pbar:
        clean_img = batch['clean_img'].to(device)
        occluded_img = batch['occluded_img'].to(device)
        binary_mask = batch['binary_mask'].to(device)
        is_clean = batch['is_clean']
        
        # Split batch into clean and occluded pairs
        # For Phase 1: Use only clean pairs (both inputs are clean)
        # For Phase 2-4: Use 50/50 split
        
        if not config['use_occluded']:
            # Phase 1: Clean only
            student_input = clean_img
            student_mask = torch.zeros_like(binary_mask)  # No mask
        else:
            # Phase 2-4: Use occluded images
            student_input = occluded_img
            student_mask = binary_mask
        
        optimizer.zero_grad()
        
        # Forward passes
        with torch.no_grad():
            teacher_embed = teacher(clean_img)  # [B, 16]
        
        student_embed_occ, attention_map_occ = student(student_input, student_mask)
        student_embed_clean, attention_map_clean = student(clean_img, None)
        
        # Downsample binary mask to 5x5 for attention regularization (matches InceptionResnetV1 feature maps)
        binary_mask_down = torch.nn.functional.adaptive_avg_pool2d(binary_mask, (5, 5))
        binary_mask_down = binary_mask_down.squeeze(1)  # [B, 5, 5]
        
        # Compute loss
        loss, loss_dict = loss_fn(
            teacher_embed_clean=teacher_embed,
            student_embed_occluded=student_embed_occ,
            student_embed_clean=student_embed_clean,
            attention_map_occluded=attention_map_occ,
            attention_map_clean=attention_map_clean,
            binary_mask_down=binary_mask_down
        )
        
        # Backward and optimize
        loss.backward()
        
        # Clip gradients: general parameters at 1.0, beta separately at 0.1
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        if config['phase'] >= 3:  # Beta is trainable in Phase 3-4
            torch.nn.utils.clip_grad_norm_([student.attention.beta], max_norm=0.1)
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss_dict['total']
        total_distill += loss_dict['distillation']
        total_consistency += loss_dict['consistency']
        total_attn_reg += loss_dict['attention_reg']
        total_attn_div += loss_dict['attention_div']
        num_batches += 1
        
        # Compute attention metrics (Phase 3-4)
        if config['phase'] >= 3 and config['lambda3'] > 0:
            with torch.no_grad():
                metrics = compute_attention_metrics(attention_map_occ, binary_mask_down)
                total_correlation += metrics['correlation']
                total_sparsity += metrics['sparsity']
                total_entropy += metrics['entropy']
                num_attention_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'dist': f"{loss_dict['distillation']:.4f}",
            'cons': f"{loss_dict['consistency']:.4f}"
        })
    
    # Average metrics
    metrics = {
        'loss': total_loss / num_batches,
        'distillation': total_distill / num_batches,
        'consistency': total_consistency / num_batches,
        'attention_reg': total_attn_reg / num_batches,
        'attention_div': total_attn_div / num_batches,
        'beta': student.get_beta_value()
    }
    
    if config['phase'] >= 3 and num_attention_batches > 0:
        metrics['attn_correlation'] = total_correlation / num_attention_batches
        metrics['attn_sparsity'] = total_sparsity / num_attention_batches
        metrics['attn_entropy'] = total_entropy / num_attention_batches
    
    return metrics


def evaluate(student, teacher, dataloader, loss_fn, device):
    """
    Evaluate on validation/test set
    
    Returns:
        dict with evaluation metrics
    """
    student.eval()
    teacher.eval()
    
    total_loss = 0.0
    total_cosine_sim = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            clean_img = batch['clean_img'].to(device)
            occluded_img = batch['occluded_img'].to(device)
            binary_mask = batch['binary_mask'].to(device)
            
            # Teacher on clean
            teacher_embed = teacher(clean_img)
            
            # Student on occluded
            student_embed, _ = student(occluded_img, binary_mask)
            
            # Cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(teacher_embed, student_embed, dim=1)
            total_cosine_sim += cos_sim.mean().item()
            num_batches += 1
    
    metrics = {
        'cosine_similarity': total_cosine_sim / num_batches
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='4-Phase Curriculum Training for Student FECNet')
    
    # Paths
    parser.add_argument('--teacher_path', type=str, default='pretrained/FECNet.pt',
                       help='Path to pretrained teacher model')
    parser.add_argument('--train_csv', type=str, default='data/dataset_pairs_train.csv',
                       help='Training dataset CSV')
    parser.add_argument('--test_csv', type=str, default='data/dataset_pairs_test.csv',
                       help='Test dataset CSV')
    parser.add_argument('--data_root', type=str, default='data',
                       help='Data root directory')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Total epochs (4 phases)')
    parser.add_argument('--batch_size', type=int, default=90,
                       help='Batch size (per spec: 90)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='DataLoader workers')
    
    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/curriculum',
                       help='Checkpoint directory')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Resume training from checkpoint path')
    
    # Logging
    parser.add_argument('--log_dir', type=str, default='logs/curriculum',
                       help='Directory for training logs')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name (default: timestamp)')
    
    # Device
    parser.add_argument('--device', type=int, default=0,
                       help='CUDA device')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Load teacher
    print("\n" + "="*70)
    print("Loading Teacher Model")
    print("="*70)
    teacher = FECNet(pretrained=False)
    teacher.load_state_dict(torch.load(args.teacher_path, map_location=device))
    teacher = teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    print("Teacher loaded and frozen")
    
    # Initialize student
    print("\n" + "="*70)
    print("Initializing Student Model")
    print("="*70)
    student = StudentFECNet(pretrained_teacher_path=args.teacher_path)
    student = student.to(device)
    
    num_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")
    print(f"Architecture: InceptionResnetV1 (1792 channels @ 5x5) + Attention + DenseNet")
    print(f"Note: Using InceptionResnetV1 instead of FaceNet NN2 (1024 @ 7x7) from paper")
    
    # Load datasets
    print("\n" + "="*70)
    print("Loading Datasets")
    print("="*70)
    train_dataset = PairedOcclusionDataset(args.train_csv, args.data_root, augment=True)
    test_dataset = PairedOcclusionDataset(args.test_csv, args.data_root, augment=False)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    # Initialize logger
    print("\n" + "="*70)
    print("Initializing Logger")
    print("="*70)
    logger = TrainingLogger(args.log_dir, args.experiment_name)
    
    # Initialize loss function
    loss_fn = MultiComponentLoss(lambda1=1.0, lambda2=0.5, lambda3=0.0, lambda4=0.01)
    loss_fn = loss_fn.to(device)
    
    # Resume from checkpoint if specified
    start_epoch = 1
    best_cosine_sim = 0.0
    
    if args.resume_from:
        print("\n" + "="*70)
        print("Resuming from Checkpoint")
        print("="*70)
        checkpoint = torch.load(args.resume_from, map_location=device)
        student.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_cosine_sim = checkpoint.get('best_cosine_sim', 0.0)
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Resuming from epoch {start_epoch}")
        print(f"Best cosine similarity so far: {best_cosine_sim:.6f}")
    
    # Training loop
    print("\n" + "="*70)
    print(f"Starting 4-Phase Curriculum Training (epochs {start_epoch}-{args.epochs})")
    print("="*70)
    
    current_phase = 0
    
    for epoch in range(start_epoch, args.epochs + 1):
        # Determine phase and configuration
        phase = determine_phase(epoch)
        config = get_phase_config(phase, epoch)
        
        # Phase transition
        if phase != current_phase:
            current_phase = phase
            print("\n" + "="*70)
            print(f"PHASE {phase}: {config['description']}")
            print(f"Epochs {config['epochs'][0]}-{config['epochs'][1]}")
            print("="*70)
            
            # Freeze/unfreeze attention
            if config['attention_frozen']:
                student.freeze_attention()
            else:
                student.unfreeze_attention()
            
            # Update loss function lambda values
            loss_fn.lambda1 = config['lambda1']
            loss_fn.lambda2 = config['lambda2']
            loss_fn.lambda3 = config['lambda3']
            loss_fn.lambda4 = config['lambda4']
            
            # Create new optimizer with phase-specific learning rates
            param_groups = [
                {'params': student.densenet.parameters(), 'lr': config['lr_densenet']},
            ]
            
            if not config['attention_frozen']:
                param_groups.extend([
                    {'params': student.attention.conv_fusion.parameters(), 'lr': config['lr_attention']},
                    {'params': student.attention.bn_fusion.parameters(), 'lr': config['lr_attention']},
                    {'params': student.attention.conv_attention.parameters(), 'lr': config['lr_attention']},
                    {'params': [student.attention.beta], 'lr': config['lr_beta']}
                ])
            
            optimizer = optim.Adam(param_groups, weight_decay=1e-4)
        
        # Update lambda3 for Phase 3 (progressive)
        if phase == 3:
            loss_fn.update_lambda3(config['lambda3'])
        
        # Train epoch
        print(f"\nEpoch {epoch}/{args.epochs} [Phase {phase}]")
        print("-" * 70)
        train_metrics = train_epoch(student, teacher, train_loader, optimizer, 
                                    loss_fn, device, config, epoch)
        
        # Log training metrics
        logger.log_train_epoch(epoch, phase, train_metrics, config)
        
        # Print metrics
        print(f"  Loss: {train_metrics['loss']:.6f}")
        print(f"  Distillation: {train_metrics['distillation']:.6f}")
        print(f"  Consistency: {train_metrics['consistency']:.6f}")
        print(f"  Beta: {train_metrics['beta']:.6f}")
        
        if phase >= 3:
            print(f"  Attention Reg: {train_metrics['attention_reg']:.6f}")
            print(f"  Attention Div: {train_metrics['attention_div']:.6f}")
            if 'attn_correlation' in train_metrics:
                corr = train_metrics['attn_correlation']
                print(f"  Attn Correlation: {corr:.4f}")
                print(f"  Attn Sparsity: {train_metrics['attn_sparsity']:.4f}")
                print(f"  Attn Entropy: {train_metrics['attn_entropy']:.4f}")
                
                # Auto-adjust lambda3 if correlation too high (>0.95)
                if corr > 0.95 and loss_fn.lambda3 > 0.01:
                    old_lambda3 = loss_fn.lambda3
                    loss_fn.lambda3 *= 0.5  # Reduce by 50%
                    print(f"  [WARNING] Correlation too high! Reducing lambda3: {old_lambda3:.4f} to {loss_fn.lambda3:.4f}")
        
        # Evaluate every 5 epochs or at phase boundaries
        if epoch % args.save_every == 0 or epoch in [10, 20, 40, 100]:
            print(f"\nEvaluating at epoch {epoch}...")
            eval_metrics = evaluate(student, teacher, test_loader, loss_fn, device)
            print(f"  Test Cosine Similarity: {eval_metrics['cosine_similarity']:.6f}")
            
            # Log evaluation metrics
            logger.log_eval_epoch(epoch, phase, eval_metrics)
            
            # Save checkpoint
            checkpoint_path = Path(args.checkpoint_dir) / f'student_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'phase': phase,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'eval_metrics': eval_metrics,
                'config': config,
                'best_cosine_sim': best_cosine_sim
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path.name}")
            
            # Save best model
            if eval_metrics['cosine_similarity'] > best_cosine_sim:
                best_cosine_sim = eval_metrics['cosine_similarity']
                best_path = Path(args.checkpoint_dir) / 'student_best.pth'
                torch.save(student.state_dict(), best_path)
                print(f"  Best model updated: {best_path.name} (sim: {best_cosine_sim:.6f})")
    
    # Final save
    final_path = Path(args.checkpoint_dir) / 'student_final.pth'
    torch.save(student.state_dict(), final_path)
    print(f"\nTraining complete! Final model saved: {final_path.name}")
    
    # Print log paths
    log_paths = logger.get_log_paths()
    print("\nTraining logs saved to:")
    print(f"  Train: {log_paths['train']}")
    print(f"  Eval:  {log_paths['eval']}")
    print(f"\nTo visualize results, run:")
    print(f"  python visualize_training_curves.py --log_dir {args.log_dir} --experiment {log_paths['experiment_name']}")
    print("="*70)


if __name__ == '__main__':
    main()

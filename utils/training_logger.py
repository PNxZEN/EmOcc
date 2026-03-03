"""
Training Logger for Curriculum Learning
Logs metrics to CSV for later visualization
"""

import csv
from pathlib import Path
from datetime import datetime


class TrainingLogger:
    """
    Logger for training and validation metrics
    
    Logs to CSV file with columns:
    - epoch, phase, loss, distillation, consistency, attention_reg, attention_div, beta
    - eval_cosine_similarity (if evaluation performed)
    - attn_correlation, attn_sparsity, attn_entropy (Phase 3-4 only)
    
    Args:
        log_dir: Directory to save logs
        experiment_name: Name of experiment (default: timestamp)
    """
    
    def __init__(self, log_dir, experiment_name=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.experiment_name = experiment_name
        self.train_log_path = self.log_dir / f"{experiment_name}_train.csv"
        self.eval_log_path = self.log_dir / f"{experiment_name}_eval.csv"
        
        # Initialize CSV files
        self._init_train_csv()
        self._init_eval_csv()
        
        print(f"[Logger] Logging to:")
        print(f"  Train: {self.train_log_path}")
        print(f"  Eval:  {self.eval_log_path}")
    
    def _init_train_csv(self):
        """Initialize training log CSV with headers"""
        with open(self.train_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'phase', 'loss', 'distillation', 'consistency',
                'attention_reg', 'attention_div', 'beta',
                'attn_correlation', 'attn_sparsity', 'attn_entropy',
                'lambda1', 'lambda2', 'lambda3', 'lambda4'
            ])
    
    def _init_eval_csv(self):
        """Initialize evaluation log CSV with headers"""
        with open(self.eval_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'phase', 'cosine_similarity'
            ])
    
    def log_train_epoch(self, epoch, phase, metrics, config):
        """
        Log training metrics for one epoch
        
        Args:
            epoch: Epoch number
            phase: Phase number (1-4)
            metrics: Dictionary with training metrics
            config: Phase configuration dictionary
        """
        with open(self.train_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                phase,
                metrics.get('loss', 0.0),
                metrics.get('distillation', 0.0),
                metrics.get('consistency', 0.0),
                metrics.get('attention_reg', 0.0),
                metrics.get('attention_div', 0.0),
                metrics.get('beta', 0.0),
                metrics.get('attn_correlation', ''),
                metrics.get('attn_sparsity', ''),
                metrics.get('attn_entropy', ''),
                config.get('lambda1', 0.0),
                config.get('lambda2', 0.0),
                config.get('lambda3', 0.0),
                config.get('lambda4', 0.0)
            ])
    
    def log_eval_epoch(self, epoch, phase, metrics):
        """
        Log evaluation metrics
        
        Args:
            epoch: Epoch number
            phase: Phase number (1-4)
            metrics: Dictionary with evaluation metrics
        """
        with open(self.eval_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                phase,
                metrics.get('cosine_similarity', 0.0)
            ])
    
    def get_log_paths(self):
        """Return paths to log files"""
        return {
            'train': str(self.train_log_path),
            'eval': str(self.eval_log_path),
            'experiment_name': self.experiment_name
        }

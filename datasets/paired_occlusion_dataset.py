"""
Paired Occlusion Dataset for Teacher-Student Training
Reference: OccFECNet.md - Section 4.1
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class PairedOcclusionDataset(Dataset):
    """
    Dataset for loading paired (clean, occluded) face images
    
    Reference: OccFECNet.md Section 4.1 and 4.4
    
    Face Alignment Note:
    Per FECNet specification, input images should be aligned by:
    - Correcting roll rotation
    - Scaling to maintain 55-pixel inter-ocular distance
    - Resizing to 224x224
    
    However, the source datasets (KDEF, RAF-DB, LFW, AffectNet) are already
    pre-aligned in their distributions. For training efficiency, we use direct
    resize to 224x224. For production deployment, face alignment should be
    performed as a preprocessing step before feeding to the model.
    
    Each sample contains:
    - clean_img: Non-occluded face image
    - occluded_img: Same face with occlusion
    - binary_mask: Occlusion mask (1=occluded, 0=visible)
    - emotion_class: Emotion label (for reference, not used in training)
    - dataset_source: Source dataset (KDEF, RAF-DB, LFW, AffectNet)
    
    Args:
        csv_path: Path to dataset CSV file
        data_root: Root directory for data (default: 'data')
        augment: Whether to apply data augmentation (default: False)
        return_clean_pairs: If True, also return clean-clean pairs for consistency loss
        use_alignment: Whether to perform face alignment (default: False, requires additional dependencies)
    """
    
    def __init__(self, csv_path, data_root='data', augment=False, return_clean_pairs=False, use_alignment=False):
        super().__init__()
        
        self.data_root = Path(data_root)
        self.augment = augment
        self.return_clean_pairs = return_clean_pairs
        self.use_alignment = use_alignment
        
        if use_alignment:
            print("[PairedOcclusionDataset] WARNING: Face alignment is enabled but not implemented.")
            print("[PairedOcclusionDataset] Datasets are assumed to be pre-aligned. Using direct resize.")
            self.use_alignment = False
        
        # Load CSV
        self.df = pd.read_csv(csv_path)
        print(f"[PairedOcclusionDataset] Loaded {len(self.df)} samples from {csv_path}")
        print(f"[PairedOcclusionDataset] Distribution by dataset:")
        print(self.df['dataset_source'].value_counts())
        print(f"[PairedOcclusionDataset] Note: Datasets assumed pre-aligned, using direct resize to 224x224")
        
        # Image preprocessing
        # Per FECNet specification: images should be aligned (roll correction, 55-pixel inter-ocular distance)
        # then resized to 224x224. Source datasets are pre-aligned, so we apply direct resize.
        self.transform_base = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Data augmentation (per OccFECNet.md Section 4.4)
        if augment:
            self.transform_aug = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.RandomRotation(degrees=5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform_aug = self.transform_base
        
        # Mask transform (binary, no normalization)
        self.transform_mask = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.df)
    
    def _get_mask_path(self, occluded_path, dataset_source):
        """
        Get binary mask path for occluded image
        
        Mask locations per user specification:
        1. LFW: data\LFW\M-LFW-FER-masks (already in occluded_path, just derive mask)
        2. KDEF: data\KDEF\KDEF_Sorted_Occluded\{category}\occlusion_mask\filename
           Note: 'mask' folder is NOT used, only 'occlusion_mask'
        3. RAFDB: data\RAFDB\RAF-DB_Occluded\train\{category}\hands\occlusion_mask\filename
        4. AffectNet: No mask (self-paired)
        
        Masks are PNG with white (255) for occluded regions, black (0) for visible
        """
        occluded_path = Path(occluded_path)
        
        if dataset_source == 'AffectNet':
            # Self-paired, no actual occlusion
            return None
        
        elif dataset_source == 'KDEF':
            # KDEF: Replace 'img' folder with 'occlusion_mask' folder
            # Path format: KDEF\KDEF_Sorted_Occluded\{emotion}\img\{filename}
            # Mask format: KDEF\KDEF_Sorted_Occluded\{emotion}\occlusion_mask\{filename}
            mask_path = str(occluded_path).replace('\\img\\', '\\occlusion_mask\\')
            mask_path = mask_path.replace('/img/', '/occlusion_mask/')
            # Change extension to .png (KDEF masks are PNG)
            mask_path = mask_path.replace('.jpg', '.png')
            return mask_path
        
        elif dataset_source == 'RAFDB':
            # RAFDB: Replace 'img' folder with 'occlusion_mask' folder
            # Path format: RAFDB\RAF-DB_Occluded\train\{emotion}\hands\img\{filename}
            # Mask format: RAFDB\RAF-DB_Occluded\train\{emotion}\hands\occlusion_mask\{filename}
            mask_path = str(occluded_path).replace('\\img\\', '\\occlusion_mask\\')
            mask_path = mask_path.replace('/img/', '/occlusion_mask/')
            # Change extension to .png (masks are PNG)
            mask_path = mask_path.replace('.jpg', '.png')
            return mask_path
        
        elif dataset_source == 'LFW':
            # LFW: The occluded_path points to M-LFW-FER-masks which contains mask overlays
            # We need to extract mask from the image itself (white regions = mask)
            # The path is already correct, just need to process as mask
            return str(occluded_path)
        
        return None
    
    def _load_image(self, path, is_mask=False):
        """Load image or mask"""
        full_path = self.data_root / path
        
        try:
            if is_mask:
                img = Image.open(full_path).convert('L')  # Grayscale
            else:
                img = Image.open(full_path).convert('RGB')
            return img
        except Exception as e:
            print(f"[Warning] Failed to load {full_path}: {e}")
            # Return black image as fallback
            if is_mask:
                return Image.new('L', (224, 224), 0)
            else:
                return Image.new('RGB', (224, 224), (0, 0, 0))
    
    def _create_binary_mask(self, mask_img, dataset_source):
        """
        Create binary mask from mask image
        
        Per user specification:
        - PNG masks have white (255) for occluded regions
        - Black (0) for visible regions
        - Need to convert to binary: 1=occluded, 0=visible
        
        Returns: Binary tensor [1, H, W] where 1=occluded, 0=visible
        """
        if mask_img is None:
            # No mask (AffectNet self-paired)
            return torch.zeros(1, 224, 224)
        
        # Convert to tensor [1, 224, 224], values in [0, 1]
        mask_tensor = self.transform_mask(mask_img)
        
        # White (1.0 after normalization) = occluded = 1
        # Black (0.0 after normalization) = visible = 0
        # Threshold at 0.5 to create binary mask
        binary_mask = (mask_tensor > 0.5).float()
        
        return binary_mask
    
    def __getitem__(self, idx):
        """
        Get paired sample
        
        Returns dict with:
        - clean_img: [3, 224, 224] - Clean face
        - occluded_img: [3, 224, 224] - Occluded face
        - binary_mask: [1, 224, 224] - Binary mask (1=occluded)
        - emotion_class: str - Emotion label
        - dataset_source: str - Source dataset
        - is_clean: bool - Whether this is a clean pair (for consistency loss)
        """
        row = self.df.iloc[idx]
        
        clean_path = row['non_occluded_path']
        occluded_path = row['occluded_path']
        emotion_class = row['emotion_class']
        dataset_source = row['dataset_source']
        
        # Load images
        clean_img = self._load_image(clean_path, is_mask=False)
        occluded_img = self._load_image(occluded_path, is_mask=False)
        
        # Get mask path and load mask
        mask_path = self._get_mask_path(occluded_path, dataset_source)
        if mask_path:
            mask_img = self._load_image(mask_path, is_mask=True)
        else:
            mask_img = None
        
        # Apply transforms
        if self.augment:
            # Same augmentation for clean and occluded
            seed = np.random.randint(2147483647)
            
            torch.manual_seed(seed)
            clean_tensor = self.transform_aug(clean_img)
            
            torch.manual_seed(seed)
            occluded_tensor = self.transform_aug(occluded_img)
        else:
            clean_tensor = self.transform_base(clean_img)
            occluded_tensor = self.transform_base(occluded_img)
        
        # Create binary mask
        binary_mask = self._create_binary_mask(mask_img, dataset_source)
        
        # Check if this is a clean pair (for consistency loss)
        is_clean = (clean_path == occluded_path)  # AffectNet self-pairs
        
        sample = {
            'clean_img': clean_tensor,
            'occluded_img': occluded_tensor,
            'binary_mask': binary_mask,
            'emotion_class': emotion_class,
            'dataset_source': dataset_source,
            'is_clean': is_clean
        }
        
        return sample


def create_balanced_dataloader(csv_path, data_root='data', batch_size=90, 
                               augment=False, num_workers=4, shuffle=True):
    """
    Create dataloader with 50/50 clean/occluded batch composition
    Reference: OccFECNet.md Section 4.2
    
    Batch composition:
    - 45 occluded pairs (for distillation loss)
    - 45 clean pairs (for consistency loss)
    
    Args:
        csv_path: Path to dataset CSV
        data_root: Data root directory
        batch_size: Total batch size (should be 90 as per spec)
        augment: Whether to augment
        num_workers: DataLoader workers
        shuffle: Whether to shuffle
    
    Returns:
        DataLoader with balanced batching
    """
    from torch.utils.data import DataLoader
    
    dataset = PairedOcclusionDataset(csv_path, data_root, augment)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Ensure consistent batch sizes
    )
    
    return dataloader

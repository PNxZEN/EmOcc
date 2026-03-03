"""
Student FECNet with Residual Spatial Attention
Reference: OccFECNet.md - Section 2.2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.inception_resnet_v1 import InceptionResnetV1
from models.densenet import DenseNet


class ResidualSpatialAttention(nn.Module):
    """
    Residual Spatial Attention Module
    
    Architecture (per OccFECNet.md Section 2.2, adapted for InceptionResnetV1):
    1. Mask downsampling (5x5) - training only [NOTE: 5x5 vs 7x7 in paper due to InceptionResnetV1]
    2. Feature-mask fusion (concatenate mask as channel if available)
    3. 1x1 conv (512 filters) + BN + ReLU
    4. 1x1 conv (1 filter) + Sigmoid -> attention map
    5. Residual connection: F_attended = F + beta * F * (1 - M_attention)
    
    Args:
        in_channels: Input feature channels (1792 for InceptionResnetV1 vs 1024 for FaceNet NN2)
        fusion_channels: Hidden dimension for fusion (512)
    """
    
    def __init__(self, in_channels=1792, fusion_channels=512):
        super().__init__()
        
        # Feature-mask fusion layers
        # Input: [B, 5, 5, 1792] or [B, 5, 5, 1793] (with mask)
        # Note: InceptionResnetV1 outputs 1792 channels @ 5x5 (vs FaceNet NN2: 1024 @ 7x7)
        self.conv_fusion = nn.Conv2d(in_channels + 1, fusion_channels, kernel_size=1, bias=True)
        self.bn_fusion = nn.BatchNorm2d(fusion_channels)
        
        # Attention map generation
        self.conv_attention = nn.Conv2d(fusion_channels, 1, kernel_size=1, bias=True)
        
        # Learnable beta parameter (attention strength)
        self.beta = nn.Parameter(torch.tensor(0.0))
        
        # Initialize to identity transformation
        self._init_identity()
    
    def _init_identity(self):
        """
        Identity initialization (per OccFECNet.md Section 2.2)
        - Conv weights: He init with scale 0.01 (small random values)
        - Conv biases: 0
        - Attention conv bias: 0 (sigmoid(0) = 0.5, neutral)
        - beta: 0.0 (attention has zero effect initially)
        """
        # Fusion conv
        nn.init.kaiming_normal_(self.conv_fusion.weight, mode='fan_out', nonlinearity='relu')
        self.conv_fusion.weight.data *= 0.01  # Scale down
        nn.init.constant_(self.conv_fusion.bias, 0)
        
        # Batch norm
        nn.init.constant_(self.bn_fusion.weight, 1)
        nn.init.constant_(self.bn_fusion.bias, 0)
        
        # Attention conv
        nn.init.kaiming_normal_(self.conv_attention.weight, mode='fan_out', nonlinearity='sigmoid')
        self.conv_attention.weight.data *= 0.01  # Scale down
        nn.init.constant_(self.conv_attention.bias, 0)
        
        # beta already initialized to 0.0 in parameter definition
    
    def forward(self, features, binary_mask=None):
        """
        Forward pass with optional binary mask
        
        Args:
            features: [B, 1792, 5, 5] - InceptionResnetV1 feature maps
                      (Note: 1792 @ 5x5 vs FaceNet NN2's 1024 @ 7x7 in paper)
            binary_mask: [B, 1, H, W] - Binary occlusion mask (1=occluded, 0=visible)
                         Only available during training, None during inference
        
        Returns:
            attended_features: [B, 1792, 5, 5] - Attention-modulated features
            attention_map: [B, 5, 5] - Attention weights (for logging/visualization)
        """
        B, C, H, W = features.shape  # [B, 1792, 5, 5]
        
        # Step 1: Mask downsampling (if available, training only)
        if binary_mask is not None:
            # Downsample mask to 5x5 using average pooling (matches InceptionResnetV1 output)
            mask_down = F.adaptive_avg_pool2d(binary_mask, (H, W))  # [B, 1, 5, 5]
        else:
            # Inference: no mask available
            mask_down = None
        
        # Step 2: Feature-mask fusion
        if mask_down is not None:
            # Training: concatenate mask as additional channel
            features_concat = torch.cat([features, mask_down], dim=1)  # [B, 1793, 5, 5]
        else:
            # Inference: pad with zeros to match expected input channels
            # This allows the same conv weights to be used
            zeros = torch.zeros(B, 1, H, W, device=features.device, dtype=features.dtype)
            features_concat = torch.cat([features, zeros], dim=1)  # [B, 1793, 5, 5]
        
        # Step 3: 1x1 convolution + BN + ReLU
        fused = self.conv_fusion(features_concat)  # [B, 512, 5, 5]
        fused = self.bn_fusion(fused)
        fused = F.relu(fused)
        
        # Step 4: Attention map generation
        attention_logits = self.conv_attention(fused)  # [B, 1, 5, 5]
        attention_map = torch.sigmoid(attention_logits)  # [B, 1, 5, 5], values in [0, 1]
        
        # Step 5: Residual attention connection
        # F_attended = F + beta * F * (1 - M_attention)
        # High attention values (close to 1) indicate occluded regions (should be downweighted)
        attended_features = features + self.beta * features * (1 - attention_map)  # [B, 1792, 5, 5]
        
        # Return both attended features and attention map (for logging)
        return attended_features, attention_map.squeeze(1)  # [B, 1792, 5, 5], [B, 5, 5]


class StudentFECNet(nn.Module):
    """
    Student FECNet with Residual Spatial Attention
    
    Architecture (per OccFECNet.md Section 2):
    - Frozen FaceNet (InceptionResnetV1 up to mixed_7a) -> 5x5x1792 features
    - Residual Spatial Attention Module (inserted here)
    - DenseNet block (5 layers, growth rate 64)
    - FC layers -> 16D L2-normalized embedding
    
    Args:
        pretrained_teacher_path: Path to teacher weights for initialization (optional)
    """
    
    def __init__(self, pretrained_teacher_path=None):
        super().__init__()
        
        # Frozen FaceNet feature extractor
        self.facenet = InceptionResnetV1(pretrained='vggface2', device='cuda').eval()
        for param in self.facenet.parameters():
            param.requires_grad = False
        
        # Residual Spatial Attention (inserted after FaceNet)
        self.attention = ResidualSpatialAttention(in_channels=1792, fusion_channels=512)
        
        # DenseNet expression layers
        self.densenet = DenseNet(
            growth_rate=64,
            block_config=[5],
            num_classes=16,
            small_inputs=True,
            efficient=True,
            num_init_features=512
        ).cuda()
        
        # Load teacher weights for DenseNet initialization
        if pretrained_teacher_path:
            self._load_teacher_weights(pretrained_teacher_path)
    
    def _load_teacher_weights(self, teacher_path):
        """
        Initialize student's DenseNet from teacher
        Per OccFECNet.md: DenseNet starts from teacher, attention starts from identity
        """
        print(f"[StudentFECNet] Loading teacher weights from: {teacher_path}")
        teacher_state = torch.load(teacher_path, map_location='cuda')
        
        # Extract DenseNet weights from teacher
        student_state = self.state_dict()
        loaded_keys = []
        
        for name, param in teacher_state.items():
            # Map teacher 'dense' to student 'densenet'
            if 'dense' in name:
                student_name = name.replace('dense', 'densenet')
                if student_name in student_state:
                    student_state[student_name].copy_(param)
                    loaded_keys.append(student_name)
        
        print(f"[StudentFECNet] Loaded {len(loaded_keys)} DenseNet parameters from teacher")
        print(f"[StudentFECNet] Attention module initialized to identity (beta=0.0)")
    
    def forward(self, x, binary_mask=None):
        """
        Forward pass
        
        Args:
            x: [B, 3, 224, 224] - Input face images
            binary_mask: [B, 1, 224, 224] - Binary occlusion mask (training only)
        
        Returns:
            embedding: [B, 16] - L2-normalized embedding
            attention_map: [B, 5, 5] - Attention weights (for monitoring)
        """
        # Frozen FaceNet feature extraction
        with torch.no_grad():
            # InceptionResnetV1 returns (logits, features) where features are [B, 1792, 5, 5]
            _, features = self.facenet(x)  # [B, 1792, 5, 5]
        
        # Residual spatial attention
        attended_features, attention_map = self.attention(features, binary_mask)
        
        # DenseNet + embedding layers
        embedding = self.densenet(attended_features)  # [B, 16], L2-normalized by DenseNet
        
        return embedding, attention_map
    
    def freeze_attention(self):
        """Freeze attention module (for Phase 1 and 2)"""
        for param in self.attention.parameters():
            param.requires_grad = False
        print("[StudentFECNet] Attention module frozen (beta and conv weights)")
    
    def unfreeze_attention(self):
        """Unfreeze attention module (for Phase 3 and 4)"""
        for param in self.attention.parameters():
            param.requires_grad = True
        print("[StudentFECNet] Attention module unfrozen (beta and conv weights)")
    
    def get_beta_value(self):
        """Get current beta value (for logging)"""
        return self.attention.beta.item()

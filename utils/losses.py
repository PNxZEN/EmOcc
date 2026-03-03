import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss3Class(nn.Module):
    """
    Triplet loss for 3-class emotion labels (positive, negative, neutral)
    Ensures embeddings from same class are closer than different classes
    """
    
    def __init__(self, margin=0.2, mining='semi-hard'):
        super().__init__()
        self.margin = margin
        self.mining = mining  # 'hard', 'semi-hard', or 'all'
    
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: [B, 16] student embeddings
            labels: [B] emotion labels (0, 1, or 2)
        
        Returns:
            loss: scalar triplet loss
        """
        batch_size = embeddings.size(0)
        
        # Compute pairwise distances
        distances = self.pairwise_distances(embeddings)  # [B, B]
        
        # Mine triplets
        if self.mining == 'semi-hard':
            loss = self.semi_hard_triplet_mining(distances, labels)
        elif self.mining == 'hard':
            loss = self.hard_triplet_mining(distances, labels)
        else:
            loss = self.all_triplet_loss(distances, labels)
        
        return loss
    
    def pairwise_distances(self, embeddings):
        """Compute pairwise L2 distances"""
        dot_product = torch.matmul(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)
        
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        distances = torch.clamp(distances, min=0.0)
        
        # Avoid NaN gradients for exact matches
        mask = torch.eq(distances, 0.0).float()
        distances = distances + mask * 1e-16
        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask)
        
        return distances
    
    def semi_hard_triplet_mining(self, distances, labels):
        """
        Semi-hard negative mining: negatives that are farther than anchor-positive,
        but within the margin
        """
        batch_size = labels.size(0)
        
        # Create masks for valid triplets
        label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]
        label_not_equal = ~label_equal
        
        # Anchor-positive distances (same label)
        anchor_positive_dist = distances.unsqueeze(2)  # [B, B, 1]
        
        # Anchor-negative distances (different label)
        anchor_negative_dist = distances.unsqueeze(1)  # [B, 1, B]
        
        # Triplet loss
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin
        
        # Mask invalid triplets
        mask = label_equal.unsqueeze(2) & label_not_equal.unsqueeze(1)  # [B, B, B]
        mask = mask.float()
        
        # Semi-hard: only consider negatives farther than positive but within margin
        semi_hard_mask = (triplet_loss > 0.0) & (triplet_loss < self.margin)
        mask = mask * semi_hard_mask.float()
        
        # Average over valid triplets
        triplet_loss = triplet_loss * mask
        num_valid_triplets = torch.sum(mask) + 1e-16
        
        loss = torch.sum(triplet_loss) / num_valid_triplets
        
        return loss
    
    def hard_triplet_mining(self, distances, labels):
        """
        Hard negative mining: hardest positive and hardest negative per anchor
        """
        batch_size = labels.size(0)
        
        # Masks
        label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        label_not_equal = ~label_equal
        
        # Hardest positive: farthest same-class sample
        masked_anchor_positive_dist = distances * label_equal.float()
        hardest_positive_dist, _ = torch.max(masked_anchor_positive_dist, dim=1)
        
        # Hardest negative: closest different-class sample
        max_anchor_negative_dist = torch.max(distances)
        masked_anchor_negative_dist = distances + max_anchor_negative_dist * (~label_not_equal).float()
        hardest_negative_dist, _ = torch.min(masked_anchor_negative_dist, dim=1)
        
        # Triplet loss
        loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
        loss = torch.mean(loss)
        
        return loss
    
    def all_triplet_loss(self, distances, labels):
        """Use all valid triplets (memory intensive)"""
        # Similar to semi-hard but include all valid triplets
        # Implementation similar to semi_hard but without semi-hard mask
        pass

# Test
if __name__ == '__main__':
    # Dummy data
    embeddings = torch.randn(32, 16)  # Batch of 32, 16D embeddings
    labels = torch.randint(0, 3, (32,))  # 3 classes (0, 1, 2)
    
    loss_fn = TripletLoss3Class(margin=0.2, mining='semi-hard')
    loss = loss_fn(embeddings, labels)
    print(f"Triplet loss: {loss.item():.4f}")
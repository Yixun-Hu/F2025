import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from typing import Optional


class DiscreteActionModel(nn.Module):
    """MobileNetV2 backbone producing discretized action logits (optionally with state).

    - Supports stacked frames: pass images shaped (B, 3*T, H, W); we pool features over T.
    - Supports concatenating a state vector before the prediction head.

    Args:
        pretrained: load ImageNet weights for the backbone.
        action_dim: number of action channels (e.g., 4).
        num_bins: number of discrete bins per action.
        action_low/high: clipping range for continuous actions.
        state_dim: optional state vector dimension to concatenate after image features.
        freeze_backbone: if True, freeze backbone feature extractor.

    Forward:
        Forward takes (img, state=None). `img` may be stacked frames (3*T channels).
        Returns logits of shape (B, action_dim, num_bins) or probabilities if
        `return_probs` is True.
    """

    def __init__(self,
                 pretrained: bool = False,
                 action_dim: int = 4,
                 num_bins: int = 11,
                 action_low: Optional[float] = -1.0,
                 action_high: Optional[float] = 1.0,
                 state_dim: int = 9,
                 freeze_backbone: bool = False,
                 ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.num_bins = int(num_bins)
        self.action_low = action_low
        self.action_high = action_high
        self.state_dim = int(state_dim)
        self.freeze_backbone = bool(freeze_backbone)

        # Load MobileNetV2 backbone. When pretrained=False this will init random weights.
        self.backbone = mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
        if self.freeze_backbone:
            for p in self.backbone.features.parameters():
                p.requires_grad = False

        in_feature_dim = self.backbone.classifier[1].in_features
        # Expose pooled feature; head will fuse state (if provided) then predict bins.
        self.backbone.classifier = nn.Identity()

        head_in = in_feature_dim + self.state_dim
        out_features = self.action_dim * self.num_bins
        self.head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(head_in, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_features),
        )

    def forward(self,
                img: torch.Tensor,
                state: Optional[torch.Tensor] = None,
                return_probs: bool = False) -> torch.Tensor:
        """Run forward pass.

        Args:
            img: images tensor of shape (batch, 3*T, H, W) or (batch, 3, H, W).
            state: optional tensor (batch, state_dim).
            return_probs: if True, return softmax probabilities over bins.

        Returns:
            logits of shape (batch, action_dim, num_bins) or probabilities if
            `return_probs` is True.
        """
        b, c, h, w = img.shape
        if c % 3 != 0:
            raise ValueError(f"Expected channels multiple of 3, got {c}")
        t = c // 3
        img_seq = img.view(b * t, 3, h, w)
        feat_seq = self.backbone(img_seq)          # [B*T, feat]
        feat_seq = feat_seq.view(b, t, -1)
        feat = feat_seq.mean(dim=1)                # temporal average pooling
        if self.state_dim > 0 and state is not None:
            feat = torch.cat([feat, state], dim=-1)
        logits = self.head(feat)
        logits = logits.view(b, self.action_dim, self.num_bins)

        if return_probs:
            return torch.softmax(logits, dim=-1)

        return logits
    
    def continuous_to_bins(self, 
                           actions: torch.Tensor,
                           ) -> torch.Tensor:
        """Convert continuous actions to integer bin indices.

        Args:
            actions: (B, A) tensor of continuous actions.
            low: minimum action value (clamp lower bound).
            high: maximum action value (clamp upper bound).
            num_bins: number of discrete bins.

        Returns:
            (B, A) LongTensor of bin indices in [0, num_bins-1].
        """
        # clamp and normalize to [0,1]
        clipped = actions.clamp(min=self.action_low, max=self.action_high)
        denom = (self.action_high - self.action_low) if (self.action_high - self.action_low) != 0 else 1.0
        norm = (clipped - self.action_low) / denom
        # map to [0, num_bins-1] and round to nearest bin
        bins = (norm * (self.num_bins - 1)).round().long()
        return bins
    
    def bins_to_continuous(self,
                           bins: torch.Tensor,
                           ) -> torch.Tensor:
        """Convert integer bin indices to continuous actions.

        Args:
            bins: (B, A) LongTensor of bin indices in [0, num_bins-1].
            low: minimum action value.
            high: maximum action value.
            num_bins: number of discrete bins.

        Returns:
            (B, A) tensor of continuous actions.
        """
        # Clamp bins then map linearly back to the continuous range.
        bins = bins.clamp(min=0, max=self.num_bins - 1).float()
        denom = (self.num_bins - 1) if (self.num_bins - 1) != 0 else 1.0
        norm = bins / denom
        return self.action_low + norm * (self.action_high - self.action_low)
    
    def output_to_executable_actions(self, output: torch.Tensor) -> torch.Tensor:
        """Convert model output logits to executable continuous actions.

        Args:
            output: (B, action_dim, num_bins) tensor of logits.

        Returns:
            (B, action_dim) tensor of continuous actions.
        """
        if output.dim() == 2:
            output = output.unsqueeze(0)
        if output.dim() != 3:
            raise ValueError(f"Expected output shape (B, action_dim, num_bins), got {tuple(output.shape)}")

        # Choose the most likely bin per action and map back to continuous space.
        best_bins = torch.argmax(output, dim=-1)  # (B, action_dim)
        actions = self.bins_to_continuous(best_bins)

        # Return CPU numpy array for easy downstream use (flatten if batch of 1).
        actions_np = actions.detach().cpu().numpy()
        if actions_np.shape[0] == 1:
            return actions_np.flatten()
        return actions_np
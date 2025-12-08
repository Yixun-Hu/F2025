import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from omegaconf import OmegaConf
from pathlib import Path
import numpy as np
from tqdm import tqdm
from typing import Optional

# Ensure we can import from the drone package
# Assuming this script is in final_project/ and drone/ is a subdirectory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from drone.datasets.dataloader import CrazyFlieILDataset
from torchvision.models import mobilenet_v2

# -----------------------------------------------------------------------------
# Model Definition (Discrete Action)
# -----------------------------------------------------------------------------

class DroneControlNet(nn.Module):
    """MobileNetV2 backbone producing discretized action logits.

    This model predicts discrete motion by outputting logits for a fixed number
    of bins per action channel. The final linear layer outputs `action_dim * num_bins`
    values which are reshaped to (batch, action_dim, num_bins).
    """

    def __init__(self,
                 cfg,
                 pretrained: bool = False,
                 action_dim: int = 4,
                 num_bins: int = 11,
                 action_low: float = -1.0,
                 action_high: float = 1.0
                 ):
        super().__init__()
        # Handle config if passed as object or kwargs
        if hasattr(cfg, 'action_dim'):
            self.action_dim = cfg.action_dim
            self.num_bins = cfg.num_bins
            self.action_low = cfg.action_low
            self.action_high = cfg.action_high
            pretrained = cfg.pretrained
        else:
            self.action_dim = int(action_dim)
            self.num_bins = int(num_bins)
            self.action_low = action_low
            self.action_high = action_high

        # Load MobileNetV2 backbone
        self.backbone = mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
        in_feature_dim = self.backbone.classifier[1].in_features

        out_features = self.action_dim * self.num_bins
        # Replace classifier with a dropout + linear that outputs logits for all bins
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_feature_dim, out_features),
        )

    def forward(self, x: torch.Tensor, return_probs: bool = False) -> torch.Tensor:
        """Run forward pass.
        Returns: logits of shape (batch, action_dim, num_bins)
        """
        logits = self.backbone(x)
        batch = logits.shape[0]
        logits = logits.view(batch, self.action_dim, self.num_bins)

        if return_probs:
            return torch.softmax(logits, dim=-1)

        return logits
    
    def continuous_to_bins(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert continuous actions to integer bin indices."""
        # clamp and normalize to [0,1]
        clipped = actions.clamp(min=self.action_low, max=self.action_high)
        denom = (self.action_high - self.action_low) if (self.action_high - self.action_low) != 0 else 1.0
        norm = (clipped - self.action_low) / denom
        # map to [0, num_bins-1] and round to nearest bin
        bins = (norm * (self.num_bins - 1)).round().long()
        return bins
    
    def bins_to_continuous(self, bins: torch.Tensor) -> torch.Tensor:
        """Convert integer bin indices to continuous actions."""
        denom = (self.action_high - self.action_low) if (self.action_high - self.action_low) != 0 else 1.0
        norm = bins.float() / (self.num_bins - 1)
        continuous = norm * denom + self.action_low
        return continuous
    
    def output_to_executable_actions(self, output: torch.Tensor) -> np.ndarray:
        """Convert model output logits to executable continuous actions (NumPy)."""
        # output: (B, action_dim, num_bins) or (action_dim, num_bins)
        if output.dim() == 2:
             output = output.unsqueeze(0)
        
        bins = torch.argmax(output, dim=-1) # (B, action_dim)
        continuous = self.bins_to_continuous(bins)
        return continuous.detach().cpu().numpy().flatten()

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

def create_dataloaders(cfg) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    
    # Initialize full dataset
    # Note: data_dir should be relative to where the script is run, or absolute
    full_dataset = CrazyFlieILDataset(
        data_dir=cfg.data_dir,
        image_size=tuple(cfg.image_size),
        normalize_states=cfg.normalize_states,
        normalize_actions=cfg.normalize_actions,
        augment=True # Augment training data
    )
    
    # Split into train/val
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle_train,
        num_workers=cfg.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers
    )
    
    return train_loader, val_loader

# -----------------------------------------------------------------------------
# Loss Function
# -----------------------------------------------------------------------------

def loss_fn(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Cross Entropy Loss for discrete actions.
    
    Args:
        outputs: (B, action_dim, num_bins) logits
        labels: (B, action_dim) integer bin indices
    """
    # CrossEntropyLoss expects (B, C, ...) and (B, ...)
    # We have multiple action dimensions. We sum the loss across dimensions.
    
    # Reshape outputs to (B*action_dim, num_bins)
    B, A, N = outputs.shape
    outputs_flat = outputs.view(B * A, N)
    
    # Reshape labels to (B*action_dim)
    labels_flat = labels.view(B * A)
    
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs_flat, labels_flat)
    
    return loss

# -----------------------------------------------------------------------------
# Main Training Loop
# -----------------------------------------------------------------------------

def main():
    # Configuration
    cfg = OmegaConf.create({
        'dataset_cfg': {
            'data_dir': 'drone/datasets/imitation_data', # Adjusted path assuming running from final_project/
            'batch_size': 32,
            'image_size': [224, 224],
            'normalize_states': False,
            'normalize_actions': False,
            'num_workers': 4,
            'shuffle_train': True,
        },
        'model_cfg': {
            'pretrained': True,
            'action_dim': 4,
            'num_bins': 11,
            'action_low': -1.0,
            'action_high': 1.0,
        },
        'training_cfg': {
            'num_epochs': 50,
            'lr': 0.001,
            'save_path': 'drone_control_model.pth'
        }
    })
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Creating dataloaders...")
    try:
        train_loader, val_loader = create_dataloaders(cfg.dataset_cfg)
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        print(f"Current working directory: {os.getcwd()}")
        print("Make sure 'drone/datasets/imitation_data' directory exists and contains trial folders.")
        return

    # Initialize model
    print("Initializing model...")
    model = DroneControlNet(cfg.model_cfg).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.training_cfg.lr)
    
    # Training Loop
    best_val_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(cfg.training_cfg.num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training_cfg.num_epochs} [Train]")
        
        for batch in pbar:
            observations = batch['observation'].to(device)
            # actions are continuous, need to convert to bins
            actions_continuous = batch['action'].to(device)
            actions_bins = model.continuous_to_bins(actions_continuous)
            
            optimizer.zero_grad()
            outputs = model(observations)
            loss = loss_fn(outputs, actions_bins)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                observations = batch['observation'].to(device)
                actions_continuous = batch['action'].to(device)
                actions_bins = model.continuous_to_bins(actions_continuous)
                
                outputs = model(observations)
                loss = loss_fn(outputs, actions_bins)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), cfg.training_cfg.save_path)
            print(f"Saved best model to {cfg.training_cfg.save_path}")

if __name__ == "__main__":
    main()


import torch
from torch.utils.data import Dataset
import sys
import os

# Ensure src module is visible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data.dataset import SolarRiskDataset

class NormalOnlyDataset(Dataset):
    def __init__(self, transform=None):
        # Load full dataset
        self.full_dataset = SolarRiskDataset(transform=transform)
        
        # Filter indices where risk score (prob) is 0.0
        self.normal_indices = []
        for idx in range(len(self.full_dataset)):
            _, score = self.full_dataset[idx]
            if score.item() == 0.0:
                self.normal_indices.append(idx)
                
        print(f"NormalOnlyDataset: Filtered {len(self.normal_indices)} healthy images from {len(self.full_dataset)} total.")

    def __len__(self):
        return len(self.normal_indices)

    def __getitem__(self, idx):
        # Map local index to global index
        global_idx = self.normal_indices[idx]
        image, _ = self.full_dataset[global_idx]
        # Autoencoder expects (Input, Target), where Target == Input
        return image, image


import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np

class SolarRiskDataset(Dataset):
    def __init__(self, root_dir='data/elpv-dataset/src/elpv_dataset/data', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data_frame = None
        
        csv_path = os.path.join(root_dir, 'labels.csv')
        # Check if dataset exists
        if not os.path.exists(csv_path):
            print(f"Warning: labels.csv not found in {root_dir}. Dataset might be empty.")
            return

        try:
            self.data_frame = pd.read_csv(csv_path, sep='\\s+', names=['name', 'prob', 'type'], header=None)
            # The official repo usually has space-separated or CSV. 
            # If reading fails (e.g. it is a proper CSV with header), fallback.
            if self.data_frame.shape[1] == 1: # Failed to parse sep
                 self.data_frame = pd.read_csv(csv_path)

            # Normalize check
            print(f"Loaded ELPV dataset with {len(self.data_frame)} samples.")
            
            # Use root_dir as base since CSV has relative paths like 'images/cell001.png'
            self.img_dir = root_dir
                 
        except Exception as e:
            print(f"Error loading labels: {e}")

    def __len__(self):
        return len(self.data_frame) if self.data_frame is not None else 0
        
    def __getitem__(self, idx):
        if self.data_frame is None:
            return torch.zeros((3, 256, 256)), torch.tensor([0.0])
            
        row = self.data_frame.iloc[idx]
        
        # Name is usually col 0
        img_name = row.iloc[0] 
        # Prob is usually col 1
        risk = float(row.iloc[1])
            
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load Image
        try:
            image = Image.open(img_path).convert('RGB')
            # If grayscale, convert to RGB for SAM
        except:
             # Fallback black image
             image = Image.new('RGB', (256, 256))
            
        # Transform
        if self.transform:
            image_tensor = self.transform(image)
        else:
            # Resize key for SAM (1024 ideally, but 256 ok for now)
            image = image.resize((256, 256))
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            
        score_tensor = torch.tensor([risk], dtype=torch.float32)
            
        return image_tensor, score_tensor

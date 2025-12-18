
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data.dataset import SolarRiskDataset
from src.models.risk_model import SolarMaintenanceModel

def train():
    # Configuration
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using Device: {DEVICE}")

    # 1. Transforms (Data Augmentation)
    print("Setting up Data Augmentation...")
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(), # Scales 0-1
        # Normalize? ResNet expects ImageNet mean/std.
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Prepare Data
    # We instantiate twice to apply different transforms
    print("Loading Datasets...")
    # Helper to get indices
    temp_ds = SolarRiskDataset()
    full_size = len(temp_ds)
    train_size = int(0.8 * full_size)
    val_size = full_size - train_size
    
    # Deterministic split
    indices = torch.randperm(full_size, generator=torch.Generator().manual_seed(42)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Train Set
    train_ds_base = SolarRiskDataset(transform=train_transform)
    train_dataset = Subset(train_ds_base, train_indices)
    
    # Val Set
    val_ds_base = SolarRiskDataset(transform=val_transform)
    val_dataset = Subset(val_ds_base, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train Size: {len(train_dataset)}, Val Size: {len(val_dataset)}")

    # 3. Initialize Model
    print("Initializing Model (ResNet50)...")
    model = SolarMaintenanceModel(model_name='resnet50')
    model.to(DEVICE)
    
    # Check for existing weights to resume
    chk_path = 'src/models/best_risk_model.pth'
    if os.path.exists(chk_path):
        print(f"Resuming: Loading weights from {chk_path}")
        model.load_state_dict(torch.load(chk_path, map_location=DEVICE))
    else:
        print("Starting training from scratch.")
    
    # Phase 1: Freeze Backbone initially
    model.freeze_backbone(True)
    
    # Optimizer & Loss
    # We need to filter parameters because some are frozen
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # 4. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        # Fine-Tuning Logic: Unfreeze after Epoch 2
        if epoch == 2:
            print("\n>>> UNFREEZING BACKBONE (Phase 2) <<<")
            model.freeze_backbone(False)
            # Re-initialize optimizer to include all parameters now
            # Use smaller LR for backbone stability
            optimizer = optim.AdamW([
                {'params': model.backbone.parameters(), 'lr': 1e-5}, # Low LR for backbone
                {'params': model.head.parameters(), 'lr': LEARNING_RATE}
            ])
            # Update scheduler to track new optimizer
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Step Scheduler
        scheduler.step(avg_val_loss)
        
        # Save Best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"Found Best Model (Val Loss {best_val_loss:.4f}). Saving...")
            torch.save(model.state_dict(), 'src/models/best_risk_model.pth')

if __name__ == '__main__':
    train()

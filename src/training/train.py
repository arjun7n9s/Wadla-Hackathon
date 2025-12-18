
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.dataset import SolarRiskDataset
from src.models.risk_model import SolarMaintenanceModel

def train():
    # Configuration
    BATCH_SIZE = 8
    EPOCHS = 5
    LEARNING_RATE = 1e-4 # Low LR for fine-tuning/heads
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using Device: {DEVICE}")

    # 1. Prepare Data
    print("Preparing Data...")
    full_dataset = SolarRiskDataset() # Uses default ELPV path
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Initialize Model
    print("Initializing Model...")
    model = SolarMaintenanceModel()
    model.to(DEVICE)
    
    # Phase 1: Freeze Backbone
    print("PHASE 1: Freezing Backbone...")
    model.freeze_backbone(True)
    
    # Optimizer & Loss
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # 3. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
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
            for images, targets in enumerate(val_loader):
                # enumerate gives index as first item
                images_t = targets[0].to(DEVICE)
                targets_t = targets[1].to(DEVICE)
                
                outputs = model(images_t)
                loss = criterion(outputs, targets_t)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Completed. Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save Best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print("Saving Best Model...")
            torch.save(model.state_dict(), 'src/models/best_risk_model.pth')

if __name__ == '__main__':
    train()

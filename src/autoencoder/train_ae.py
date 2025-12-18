
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.autoencoder.model import ConvAutoencoder
from src.autoencoder.dataset import NormalOnlyDataset

def train_ae():
    # Config
    BATCH_SIZE = 32
    EPOCHS = 20 # Autoencoders often need more, but start with 20
    LEARNING_RATE = 1e-3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using Device: {DEVICE}")

    # 1. Transforms
    # Autoencoders generated blurry outputs if aug is too heavy, but flipping is safe.
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        # Normalize to 0-1 for Sigmoid output? 
        # Standard ImgNet mean/std results in approx -2 to 2 range.
        # If model ends with Sigmoid, target must be 0-1.
        # So we SKIP normalize or use a custom one.
        # Let's stick to simple 0-1 range (ToTensor) and Sigmoid.
    ])
    
    # 2. Data
    print("Preparing Normal-Only Dataset...")
    full_ds = NormalOnlyDataset(transform=train_transform)
    
    train_size = int(0.9 * len(full_ds)) # 90/10 split since we have fewer normal images
    val_size = len(full_ds) - train_size
    train_dataset, val_dataset = random_split(full_ds, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Model
    model = ConvAutoencoder().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # 4. Loop
    print(f"Starting Training on {len(train_dataset)} healthy images...")
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for images, targets in train_loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            
            # Denoising: Add noise to input, target remains clean
            noise_factor = 0.1
            noisy_images = images + noise_factor * torch.randn_like(images)
            noisy_images = torch.clamp(noisy_images, 0., 1.)
            
            optimizer.zero_grad()
            outputs = model(noisy_images) # Input is noisy
            loss = criterion(outputs, targets) # Target is clean
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(DEVICE)
                targets = targets.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), 'src/autoencoder/best_ae.pth')
            
    print("Training Complete. Best AE saved.")

if __name__ == '__main__':
    train_ae()

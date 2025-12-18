
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.autoencoder.model import ConvAutoencoder
from src.data.dataset import SolarRiskDataset

def evaluate_ae():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using Device: {DEVICE}")
    
    # 1. Load Model
    model = ConvAutoencoder().to(DEVICE)
    model.load_state_dict(torch.load('src/autoencoder/best_ae.pth', map_location=DEVICE))
    model.eval()
    criterion = nn.MSELoss(reduction='none') # Pixel-wise loss
    
    # 2. Load Evaluation Data (Mixed Normal & Defective)
    # We want to see if MSE > Threshold separates them.
    from torchvision import transforms
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    full_ds = SolarRiskDataset(transform=val_transform)
    loader = DataLoader(full_ds, batch_size=32, shuffle=False)
    
    normal_errors = []
    defective_errors = []
    
    print("Running Inference over full dataset...")
    
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(DEVICE)
            # targets from dataset are (B, 1) probability. 
            # prob == 0 is normal. prob > 0 is defective.
            
            recons = model(images)
            
            # Reconstruction Error per image: Mean over (C, H, W)
            loss_per_pixel = (recons - images) ** 2
            loss_per_image = loss_per_pixel.mean(dim=[1, 2, 3]) # (B, )
            
            # CPU
            batch_losses = loss_per_image.cpu().numpy()
            targets_np = targets.numpy().flatten()
            
            for i in range(len(batch_losses)):
                error = batch_losses[i]
                risk = targets_np[i]
                
                if risk == 0.0:
                    normal_errors.append(error)
                else:
                    defective_errors.append(error)
                    
    normal_errors = np.array(normal_errors)
    defective_errors = np.array(defective_errors)
    
    print("-" * 30)
    print("ANOMALY DETECTION REPORT (Refined)")
    print("-" * 30)
    print(f"Normal Samples:    {len(normal_errors)}")
    print(f"Defective Samples: {len(defective_errors)}")
    print(f"Normal MSE (Avg):  {normal_errors.mean():.6f} +/- {normal_errors.std():.6f}")
    print(f"Defect MSE (Avg):  {defective_errors.mean():.6f} +/- {defective_errors.std():.6f}")
    
    # Adaptive Threshold
    # Mean + 2*Std of Normal Data
    threshold = normal_errors.mean() + 2 * normal_errors.std()
    print(f"\nAdaptive Threshold (Mean + 2*Std): {threshold:.6f}")
    
    detected = np.sum(defective_errors > threshold)
    false_alarms = np.sum(normal_errors > threshold)
    
    # Binary Classification Metrics
    tp = detected
    fp = false_alarms
    tn = len(normal_errors) - fp
    fn = len(defective_errors) - tp
    
    accuracy = (tp + tn) / (len(normal_errors) + len(defective_errors))
    
    print(f"Defects Detected (TP): {detected} ({detected/len(defective_errors)*100:.1f}%)")
    print(f"False Alarms (FP):     {false_alarms} ({false_alarms/len(normal_errors)*100:.1f}%)")
    print(f"Overall Accuracy:      {accuracy*100:.2f}%")
    print("-" * 30)
    
    # Save Heatmap Examples (Top 3 Defect, Top 3 Normal)
    print("Generating Heatmaps...")
    # (Re-run a small batch or just grabbing from last batch would be efficient, but let's keep it simple)
    # We'll just visualize the last batch processed in the loop for demonstration
    
    # Normalize error map for visualization
    def save_heatmap(img_tensor, recon_tensor, filename):
        mse = (img_tensor - recon_tensor) ** 2
        mse = mse.mean(dim=0).cpu().numpy() # (H, W)
        
        plt.figure(figsize=(10, 3))
        plt.subplot(1, 3, 1)
        plt.imshow(img_tensor.permute(1, 2, 0).cpu().numpy())
        plt.title("Original")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(recon_tensor.permute(1, 2, 0).cpu().detach().numpy())
        plt.title("Reconstructed")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(mse, cmap='hot')
        plt.title("Error Heatmap")
        plt.axis('off')
        
        plt.savefig(filename)
        plt.close()

    # Visualize last batch items
    if len(images) > 0:
        # Save a defective example if present (Prob > 0)
        def_indices = (targets > 0).nonzero(as_tuple=True)[0]
        if len(def_indices) > 0:
            idx = def_indices[0]
            save_heatmap(images[idx], recons[idx], 'src/autoencoder/heatmap_defective.png')
            print("Saved src/autoencoder/heatmap_defective.png")
            
        # Save a normal example
        norm_indices = (targets == 0).nonzero(as_tuple=True)[0]
        if len(norm_indices) > 0:
            idx = norm_indices[0]
            save_heatmap(images[idx], recons[idx], 'src/autoencoder/heatmap_normal.png')
            print("Saved src/autoencoder/heatmap_normal.png")

if __name__ == '__main__':
    evaluate_ae()

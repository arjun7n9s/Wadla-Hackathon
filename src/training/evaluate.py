
import os
import torch
import sys
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.dataset import SolarRiskDataset
from src.models.risk_model import SolarMaintenanceModel

def evaluate():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load Model
    model = SolarMaintenanceModel()
    try:
        model.load_state_dict(torch.load('src/models/best_risk_model.pth'))
        print("Loaded best model.")
    except:
        print("Model file not found, running with random weights.")
        
    model.to(DEVICE)
    model.eval()
    
    # Data
    dataset = SolarRiskDataset() 
    # Use subset for evaluation speed if needed, or full
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    predictions = []
    ground_truths = []
    
    print("Evaluating...")
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            output = model(images)
            
            predictions.append(output.item())
            ground_truths.append(targets.item())
            
    # Metrics
    predictions = torch.tensor(predictions)
    ground_truths = torch.tensor(ground_truths)
    
    mae = torch.mean(torch.abs(predictions - ground_truths))
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    
    # Simple Plot
    plt.figure(figsize=(10, 5))
    plt.plot(ground_truths.numpy(), label='Ground Truth Risk', marker='o')
    plt.plot(predictions.numpy(), label='Predicted Risk', marker='x')
    plt.title(f'Solar Panel Risk Score Prediction (MAE: {mae:.4f})')
    plt.legend()
    plt.savefig('evaluation_plot.png')
    print("Saved evaluation plot to evaluation_plot.png")

if __name__ == '__main__':
    evaluate()

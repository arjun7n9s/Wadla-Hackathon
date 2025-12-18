
import torch
import torch.onnx
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.autoencoder.model import ConvAutoencoder

def export_ae():
    print("Loading Autoencoder...")
    model = ConvAutoencoder()
    try:
        model.load_state_dict(torch.load('src/autoencoder/best_ae.pth', map_location='cpu'))
        print("Loaded weights.")
    except:
        print("Warning: Weights not found, exporting initialized model.")
        
    model.eval()
    
    dummy_input = torch.randn(1, 3, 256, 256)
    output_path = "src/autoencoder/solar_ae.onnx"
    
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['reconstruction'],
        dynamic_axes={'input': {0: 'batch_size'}, 'reconstruction': {0: 'batch_size'}}
    )
    print("Export success.")

if __name__ == '__main__':
    export_ae()

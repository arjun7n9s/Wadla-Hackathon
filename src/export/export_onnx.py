
import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.risk_model import SolarMaintenanceModel

def export_model():
    # 1. Load PyTorch Model
    print("Loading PyTorch model...")
    model = SolarMaintenanceModel()
    try:
        model.load_state_dict(torch.load('src/models/best_risk_model.pth'))
        print("Loaded weights from best_risk_model.pth")
    except:
        print("Warning: Weights not found. Exporting initialized model.")
    
    model.eval()
    
    # 2. Prepare Dummy Input
    # SAM input is typically 3x1024x1024 for best results, but we can make it dynamic or smaller
    dummy_input = torch.randn(1, 3, 1024, 1024)
    
    # 3. Export to ONNX
    output_path = "solar_risk_model.onnx"
    print(f"Exporting to {output_path}...")
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=17, # High opset for recent ops
            do_constant_folding=True,
            input_names=['input_image'],
            output_names=['risk_score'],
            dynamic_axes={'input_image': {0: 'batch_size', 2: 'height', 3: 'width'},
                          'risk_score': {0: 'batch_size'}}
        )
        print("Export successful!")
    except Exception as e:
        print(f"Export failed: {e}")
        return

    # 4. Verify ONNX Model
    print("Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    # 5. Run Inference Comparison
    print("Running ONNX Inference...")
    ort_session = ort.InferenceSession(output_path)
    
    # Compute PyTorch Output
    with torch.no_grad():
        torch_out = model(dummy_input).numpy()
        
    # Compute ONNX Output
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_out = ort_session.run(None, ort_inputs)[0]
    
    # Compare
    print(f"PyTorch Output: {torch_out[0][0]:.4f}")
    print(f"ONNX Output:    {ort_out[0][0]:.4f}")
    
    np.testing.assert_allclose(torch_out, ort_out, rtol=1e-03, atol=1e-05)
    print("Verification Passed: PyTorch and ONNX outputs match!")

if __name__ == '__main__':
    export_model()

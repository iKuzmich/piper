import os
import sys
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

def apply_dynamic_quantization(model_path):
    if not os.path.exists(model_path):
        print(f"Error: File {model_path} not found.")
        return

    # Generate the new filename
    base_name, ext = os.path.splitext(model_path)
    output_path = f"{base_name}dq{ext}"

    print(f"Quantizing {model_path} dynamically...")
    
    # Apply dynamic quantization
    # We use QInt8 for weights as it's generally best for CPU performance
    quantize_dynamic(
        model_input=model_path,
        model_output=output_path,
        weight_type=QuantType.QInt8
    )

    print(f"Success! Saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 dynamic_quant.py <path_to_model.onnx>")
    else:
        apply_dynamic_quantization(sys.argv[1])
import os
import sys
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader, QuantFormat

class PiperCalibrationDataReader(CalibrationDataReader):
    """Provides realistic phoneme sequences for Piper calibration."""
    def __init__(self, model_path):
        session = ort.InferenceSession(model_path)
        self.inputs = session.get_inputs()
        
        # We simulate 5-10 'sentences' of varying lengths
        # In a real Piper model, inputs are usually: 
        # [input_ids, input_lengths, scales]
        self.data_list = []
        for _ in range(10):
            seq_len = np.random.randint(10, 50)
            # Input 0: Phoneme IDs (int64)
            phonemes = np.random.randint(0, 50, (1, seq_len)).astype(np.int64)
            # Input 1: Lengths (int64)
            lengths = np.array([seq_len]).astype(np.int64)
            # Input 2: Scales (float32) - Noise scale, length scale, etc.
            scales = np.array([0.667, 1.0, 0.8], dtype=np.float32)
            
            self.data_list.append({
                self.inputs[0].name: phonemes,
                self.inputs[1].name: lengths,
                self.inputs[2].name: scales
            })
        self.enum_data = iter(self.data_list)

    def get_next(self):
        return next(self.enum_data, None)

def apply_static_quantization(model_path):
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    base_name, ext = os.path.splitext(model_path)
    output_path = f"{base_name}sq{ext}"
    
    dr = PiperCalibrationDataReader(model_path)

    print(f"Applying Static Quantization to {model_path}...")
    
    # Static quantization with per-channel weight quantization for better accuracy
    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=dr,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        extra_options={'per_channel': True}
    )

    print(f"Success! Static model saved as: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 static_quant.py <model.onnx>")
    else:
        apply_static_quantization(sys.argv[1])
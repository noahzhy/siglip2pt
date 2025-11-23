import onnx
from onnxsim import simplify
import os

# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------
INPUT_PATH = "siglip_vision.onnx"       # original exported model
OUTPUT_PATH = "siglip_vision_sim.onnx"  # simplified model

def simplify_model():
    print(f"Reading model: {INPUT_PATH} ...")
    
    # 1. Load ONNX model
    try:
        model = onnx.load(INPUT_PATH)
    except FileNotFoundError:
        print(f"Error: file not found: {INPUT_PATH}")
        return

    # 2. Run simplification
    # `simplify` returns two values: the simplified model object and a
    # boolean indicating whether validation passed.
    # dynamic_input_shape=True may be safer for complex dynamic shapes,
    # but default parameters are usually sufficient for standard ViT models.
    print("Running onnxsim for graph optimization...")
    model_simp, check = simplify(model)

    # 3. Validation
    if not check:
        print("Warning: onnxsim validation failed! The generated model may be invalid.")
    else:
        print("onnxsim validation passed.")

    # 4. Save and compare file sizes
    onnx.save(model_simp, OUTPUT_PATH)
    original_size = os.path.getsize(INPUT_PATH) / (1024 * 1024)
    sim_size = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    
    print("-" * 30)
    print(f"Original model size: {original_size:.2f} MB")
    print(f"Simplified model size: {sim_size:.2f} MB")
    print(f"Model saved to: {OUTPUT_PATH}")
    print("-" * 30)


if __name__ == "__main__":
    simplify_model()

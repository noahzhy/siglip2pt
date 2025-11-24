import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SiglipModel
import onnx

try:
    from onnxsim import simplify
except Exception:
    simplify = None


# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------
MODEL_PATH = "./siglip_output/final_model"  # or your fine-tuned model path
ONNX_PATH = "siglip_vision.onnx"
SIMPLIFIED_ONNX_PATH = "siglip_vision_sim.onnx"
OPSET_VERSION = 18  # Recommend >= 14


# ----------------------------------------------------------------
# Wrapper module
# ----------------------------------------------------------------
# This wrapper:
# 1) calls the HF Siglip model to get raw image features (tensor output),
# 2) applies L2 normalization so exported embeddings are ready-to-use.
class SiglipVisionExport(nn.Module):
    def __init__(self, model: SiglipModel):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Extract visual features using the SigLIP model's image head
        features = self.model.get_image_features(pixel_values)

        # Apply L2 normalization on the last dim so outputs are normalized
        return F.normalize(features, p=2, dim=-1)


def export_onnx(model_path: str = MODEL_PATH,
                onnx_path: str = ONNX_PATH,
                opset_version: int = OPSET_VERSION,
                batch_size: int = 8,
                height: int = 224,
                width: int = 224):
    """Export the SigLIP vision model to an ONNX file.

    The exported model returns a single tensor of normalized image embeddings.
    """
    print(f"Loading PyTorch model from: {model_path} ...")
    hf_model = SiglipModel.from_pretrained(model_path)
    hf_model.eval()

    dummy_input = torch.randn(batch_size, 3, height, width)

    if batch_size > 1:
        print("Note: exporting with dynamic batch size ...")
        dynamic_axes = {
            "pixel_values": {0: "batch_size"},
            "image_embeds": {0: "batch_size"}
        }
    else:
        dynamic_axes = None

    print(f"Exporting ONNX to {onnx_path} (opset={opset_version}) ...")
    torch.onnx.export(
        SiglipVisionExport(hf_model),
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["image_embeds"],
        dynamic_axes=dynamic_axes
    )

    print(f"ONNX export complete: {onnx_path}")


def simplify_onnx(input_path: str = ONNX_PATH, output_path: str = SIMPLIFIED_ONNX_PATH):
    """Simplify an ONNX model using onnx-simplifier and save the result.

    If onnx-simplifier is not installed, this function will inform the user.
    """
    if simplify is None:
        print("onnx-simplifier (onnxsim) is not available. Skipping simplification.")
        return

    print(f"Reading model: {input_path} ...")
    try:
        model = onnx.load(input_path)
    except FileNotFoundError:
        print(f"Error: file not found: {input_path}")
        return

    print("Running onnxsim for graph optimization...")
    model_simp, check = simplify(model)

    if not check:
        print("Warning: onnxsim validation failed! The simplified model may be invalid.")
    else:
        print("onnxsim validation passed.")

    onnx.save(model_simp, output_path)
    original_size = os.path.getsize(input_path) / (1024 * 1024)
    sim_size = os.path.getsize(output_path) / (1024 * 1024)

    print("-" * 30)
    print(f"Original model size: {original_size:.2f} MB")
    print(f"Simplified model size: {sim_size:.2f} MB")
    print(f"Simplified model saved to: {output_path}")
    print("-" * 30)


def validate_onnx(onnx_path: str = ONNX_PATH, batch_size: int = 1, height: int = 224, width: int = 224):
    """Optionally run a quick inference using onnxruntime to validate the ONNX model."""
    try:
        import onnxruntime as ort
        import numpy as np
    except Exception:
        print("onnxruntime is not installed. Skipping runtime validation.")
        return

    print(f"Validating ONNX model with onnxruntime: {onnx_path} ...")
    session = ort.InferenceSession(onnx_path)
    dummy = np.random.randn(batch_size, 3, height, width).astype(np.float32)
    ort_input = {session.get_inputs()[0].name: dummy}
    ort_output = session.run(None, ort_input)[0]
    print(f"Validation output shape: {ort_output.shape}")
    print(f"First 5 values of first embedding: {ort_output[0][:5]}")


def main():
    parser = argparse.ArgumentParser(description="Export SigLIP vision model to ONNX and simplify it.")
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--onnx_path",  default=ONNX_PATH)
    parser.add_argument("--sim_path",   default=SIMPLIFIED_ONNX_PATH)
    parser.add_argument("--opset",  type=int, default=OPSET_VERSION)
    parser.add_argument("--batch",  type=int, default=1)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width",  type=int, default=224)
    parser.add_argument("--no_simplify", action="store_true", help="Skip ONNX simplification step")
    parser.add_argument("--validate", action="store_true", help="Run onnxruntime validation after export/simplify")

    args = parser.parse_args()

    export_onnx(model_path=args.model_path,
                onnx_path=args.onnx_path,
                opset_version=args.opset,
                batch_size=args.batch,
                height=args.height,
                width=args.width)

    if not args.no_simplify:
        simplify_onnx(input_path=args.onnx_path, output_path=args.sim_path)

    if args.validate:
        # Validate simplified model if available, else validate export
        validate_target = args.sim_path if (not args.no_simplify) else args.onnx_path
        validate_onnx(onnx_path=validate_target, batch_size=1, height=args.height, width=args.width)


if __name__ == "__main__":
    main()

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

try:
    import onnx
    from onnxsim import simplify
except ImportError:
    onnx = None
    simplify = None


MODEL_PATH = "./siglip_output/final_model"
ONNX_PATH = "siglip2_naflex.onnx"
SIMPLIFIED_ONNX_PATH = "siglip2_naflex_simplified.onnx"
OPSET_VERSION = 18
DEVICE = "cpu"


# -------------------------------------------------------------
# 1. Processor
# -------------------------------------------------------------
processor = AutoProcessor.from_pretrained(MODEL_PATH)


# -------------------------------------------------------------
# 2. Wrapper ← 接受 pixel_values (tensor)，不要 dict
# -------------------------------------------------------------
class SiglipVisionWrapper(nn.Module):
    def __init__(self, model_path, device="cpu"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.model.eval()

    def forward(self, pixel_values):    # <-- 注意这里只收 tensor
        feats = self.model.get_image_features(pixel_values=pixel_values)
        feats = F.normalize(feats, p=2, dim=-1)
        return feats


model = SiglipVisionWrapper(MODEL_PATH, DEVICE)


# -------------------------------------------------------------
# 3. Test the model (optional)
# -------------------------------------------------------------
def test_model():
    dummy_image = torch.randn(1, 3, 384, 384)
    inputs = processor(images=dummy_image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)
    print("pixel_values:", pixel_values.shape)

    # Test run
    out = model(pixel_values)
    print("output:", out.shape)
    return pixel_values



# -------------------------------------------------------------
# 3. Export ONNX
# -------------------------------------------------------------
def export_onnx(model_path=MODEL_PATH,
                onnx_path=ONNX_PATH,
                opset_version=OPSET_VERSION,
                batch=1,
                height=384,
                width=384,
                device=DEVICE):

    print(f"Loading model from {model_path} ...")
    model = SiglipVisionWrapper(model_path, device).to(device)

    # ------------------------------
    # Dummy input: tensor only, shape [B,3,H,W]
    # ------------------------------
    dummy_image = torch.randn(batch, 3, height, width)
    inputs = processor(images=dummy_image, return_tensors="pt")
    dummy = inputs["pixel_values"].to(device)

    dynamic_axes = {
        "pixel_values": {0: "batch_size"},
        "image_embeds": {0: "batch_size"}
    }

    print(f"Exporting ONNX to {onnx_path} ...")
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["pixel_values"],
        output_names=["image_embeds"],
        dynamic_axes=dynamic_axes,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
    )
    print("Export finished.")


# -------------------------------------------------------------
# 4. Simplify ONNX
# -------------------------------------------------------------
def simplify_onnx(input_path=ONNX_PATH, output_path=SIMPLIFIED_ONNX_PATH):
    if simplify is None:
        print("onnxsim not installed. Skipping simplification.")
        return

    print("Simplifying ONNX...")
    model = onnx.load(input_path)
    simp, check = simplify(model)
    if not check:
        print("Warning: simplified ONNX check failed.")
    onnx.save(simp, output_path)
    print(f"Simplified model saved: {output_path}")


# -------------------------------------------------------------
# 5. Validate ONNX
# -------------------------------------------------------------
def validate_onnx(path, height=384, width=384):
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("onnxruntime not installed. Skipping validation.")
        return

    print(f"Validating ONNX model: {path} ...")
    sess = ort.InferenceSession(path)
    dummy = np.random.randn(1, 3, height, width).astype(np.float32)
    outputs = sess.run(None, {"pixel_values": dummy})
    print("Output shape:", outputs[0].shape)
    print("Sample first 5 values:", outputs[0][0][:5])


# -------------------------------------------------------------
# 6. CLI
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--onnx_path", default=ONNX_PATH)
    parser.add_argument("--sim_path", default=SIMPLIFIED_ONNX_PATH)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--opset", type=int, default=OPSET_VERSION)
    parser.add_argument("--no_simplify", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    export_onnx(args.model_path, args.onnx_path, args.opset,
                args.batch, args.height, args.width)

    if not args.no_simplify:
        simplify_onnx(args.onnx_path, args.sim_path)

    if args.validate:
        validate_onnx(args.sim_path if not args.no_simplify else args.onnx_path,
                      args.height, args.width)


if __name__ == "__main__":
    main()

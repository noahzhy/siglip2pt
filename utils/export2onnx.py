import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor
import onnx

try:
    from onnxsim import simplify
except Exception:
    simplify = None


# -------------------------------------------------------------
# 1. Config
# -------------------------------------------------------------
MODEL_PATH = "./siglip_output/final_model"
ONNX_PATH = "siglip_vision.onnx"
SIMPLIFIED_ONNX_PATH = "siglip_vision_sim.onnx"
OPSET_VERSION = 18
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(MODEL_PATH)


# -------------------------------------------------------------
# 2. Wrapper (Vision model + L2 normalize)
# -------------------------------------------------------------
class SiglipVisionWrapper(nn.Module):
    def __init__(self, model_path, device="cpu"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.model.eval()

    # 直接接收 tensor，不用 processor
    def forward(self, inputs):
        feats = self.model.get_image_features(**inputs)
        feats = F.normalize(feats, p=2, dim=-1)
        return feats


model = SiglipVisionWrapper(MODEL_PATH, DEVICE)
# model = AutoModel.from_pretrained(MODEL_PATH)
image = torch.randn(8, 3, 512, 512).to(DEVICE)
image = processor(images=image, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
print(image["pixel_values"].shape)
out = model(image)
print(out.shape)

output_onnx_path = "siglip2_naflex.onnx"

torch.onnx.export(
    model,
    image,
    output_onnx_path,
    input_names=["pixel_values"],
    # output_names=["image_embeds"], # Or other relevant output names
)

quit()

# image = torch.randn(8, 3, 512, 512)
batch = processor(images=image, return_tensors="pt").to(DEVICE)
pixel_values = batch["pixel_values"]
print(pixel_values.shape)
# out = model(pixel_values)
torch.onnx.export(model, pixel_values, ONNX_PATH, opset_version=OPSET_VERSION)

# torch.onnx.export(
#     model,
#     image,
#     ONNX_PATH,
#     # # input_names=["pixel_values"],
#     # # output_names=["image_embeds"],
#     # # dynamic_axes=dynamic_axes,
#     # export_params=True,
#     opset_version=OPSET_VERSION,
#     # do_constant_folding=True,
# )

quit()


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
    dummy = torch.randn(batch, 3, height, width, device=device)
    dummy = processor(images=dummy, return_tensors="pt").to(device)

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

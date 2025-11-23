import torch
from PIL import Image
from transformers import SiglipProcessor, SiglipModel
import torch.nn.functional as F
import numpy as np

# ----------------------------------------------------------------
# 1. Configure paths and device
# ----------------------------------------------------------------
# This can be a HuggingFace model ID or a local path from your last training run
MODEL_PATH = "./siglip_output/final_model" 

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model: {MODEL_PATH} (device: {device})...")

# ----------------------------------------------------------------
# 2. Load model and processor
# ----------------------------------------------------------------
# Note: we can load SiglipModel directly; it contains the vision encoder and projection head
model = SiglipModel.from_pretrained(MODEL_PATH).to(device)
processor = SiglipProcessor.from_pretrained(MODEL_PATH)

# Set to evaluation mode (disable dropout, etc.)
model.eval()

# ----------------------------------------------------------------
# 3. Define embedding extraction function
# ----------------------------------------------------------------
def get_image_embeddings(image_paths, normalize=True):
    """
    Input: list of image paths. Returns embeddings as a NumPy array.
    """
    images = []
    for path in image_paths:
        # Open image and convert to RGB (avoid PNG alpha channel issues)
        images.append(Image.open(path).convert("RGB"))

    # Preprocess: resize, normalize, convert to tensor
    inputs = processor(images=images, return_tensors="pt").to(device)

    # Inference
    with torch.no_grad():
        # `get_image_features` runs the vision model and projection head
        image_features = model.get_image_features(**inputs)

    # Normalize (optional but recommended)
    # CLIP/SigLIP features are typically normalized so dot product == cosine similarity
    if normalize:
        image_features = F.normalize(image_features, p=2, dim=-1)

    # Move to CPU and convert to numpy for storage in vector DBs (e.g., Milvus, Faiss)
    return image_features.cpu().numpy()

# ----------------------------------------------------------------
# 4. Run example
# ----------------------------------------------------------------
if __name__ == "__main__":
    img_paths = ["coke.jpg", "coke.jpg"]

    print("Starting feature extraction...")
    embeddings = get_image_embeddings(img_paths)
    
    print(f"Feature extraction succeeded, shape: {embeddings.shape}")
    # SigLIP SO400M output dimension is usually 1152
    
    # Compute similarity between the two images
    # Since features are normalized, dot product == cosine similarity
    similarity = embeddings[0] @ embeddings[1].T
    print(f"Similarity between image 1 and image 2: {similarity:.4f}")

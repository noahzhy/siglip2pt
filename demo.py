import torch
from transformers import AutoModel, AutoProcessor, pipeline
from transformers.image_utils import load_image


def get_image_embeddings(ckpt: str, image_path: str) -> torch.Tensor:
    """Extract image embeddings using SIGLIP2 model."""
    model = AutoModel.from_pretrained(ckpt, device_map="auto").eval()
    processor = AutoProcessor.from_pretrained(ckpt)
    image = load_image(image_path)
    inputs = processor(images=[image], return_tensors="pt").to(model.device)
    # input shapes
    print(f"Input pixel values shape: {inputs['pixel_values'].shape}")
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def zero_shot_classification(ckpt: str, image_path: str, labels: list[str]) -> list[dict]:
    """Perform zero-shot image classification."""
    classifier = pipeline(model=ckpt, task="zero-shot-image-classification")
    image = load_image(image_path)
    outputs = classifier(image, labels)
    for output in outputs:
        print(f"{output['label']}: {output['score']:.4f}")
    return outputs


def main():
    ckpt = "google/siglip2-so400m-patch16-naflex"
    ckpt = "google/siglip2-base-patch16-naflex"
    image_path = "data/000000000285.jpg"
    labels = ["2 cats", "a plane", "a bear"]

    embeddings = get_image_embeddings(ckpt, image_path)
    print("\n" + "="*50 + "\n")
    results = zero_shot_classification(ckpt, image_path, labels)


if __name__ == "__main__":
    main()

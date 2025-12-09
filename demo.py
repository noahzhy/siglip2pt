import torch
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image

# load the model and processor
ckpt = "google/siglip2-so400m-patch16-naflex"
model = AutoModel.from_pretrained(ckpt, device_map="auto").eval()
processor = AutoProcessor.from_pretrained(ckpt)

# load the image
image = load_image("data/000000000285.jpg")
inputs = processor(images=[image], return_tensors="pt").to(model.device)

# run infernece
with torch.no_grad():
    image_embeddings = model.get_image_features(**inputs)    

print(image_embeddings.shape)


from transformers import pipeline

# load pipeline
ckpt = "google/siglip2-so400m-patch16-naflex"
image_classifier = pipeline(model=ckpt, task="zero-shot-image-classification")

# load image and candidate labels
image = load_image("data/000000000285.jpg")
candidate_labels = ["2 cats", "a plane", "a bear"]

# run inference
outputs = image_classifier(image, candidate_labels)
print(outputs)

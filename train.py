import os, sys

import yaml
import torch
import torch.nn as nn
from PIL import Image
from transformers import SiglipProcessor, SiglipModel, Trainer, TrainingArguments
from datasets import Dataset


# load yaml from local file
cfg = yaml.safe_load(open("config.yaml", "r"))


# ----------------------------------------------------------------
# 2. Configuration and loading
# ----------------------------------------------------------------
# MODEL_ID = "google/siglip-so400m-patch14-384" # or the corresponding siglip2 checkpoint
MODEL_ID = "google/siglip2-base-patch16-224"
processor = SiglipProcessor.from_pretrained(MODEL_ID)

# Instantiate our custom model
model = SiglipForFineTuning(MODEL_ID)

# ----------------------------------------------------------------
# 3. Prepare dummy dataset
# ----------------------------------------------------------------
def create_dummy_dataset():
    data = []
    for i in range(32): # create 32 samples
        # create 32 samples
        # random color image
        img = Image.new('RGB', (224, 224), color=(i*5 % 255, (i*10)%255, 150))
        text = f"This is a photo of color id {i}"
        data.append({"image": img, "text": text})
    return Dataset.from_list(data)

dataset = create_dummy_dataset()

# ----------------------------------------------------------------
# 4. Data collation (collate function)
# ----------------------------------------------------------------
def collate_fn(batch):
    images = [x['image'] for x in batch]
    texts = [x['text'] for x in batch]
    
    inputs = processor(
        text=texts,
        images=images,
        padding="max_length",
        truncation=True,
        max_length=32,
        return_tensors="pt"
    )
    return inputs

# ----------------------------------------------------------------
# 5. Training arguments
# ----------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./siglip_output",
    per_device_train_batch_size=16,  # The larger the better if VRAM allows
    num_train_epochs=5,
    learning_rate=5e-6,             # SigLIP is also fine-tuning, so use a low learning rate
    logging_steps=1,
    remove_unused_columns=False,    # Must be False, otherwise the image field will be filtered out
    save_strategy="no",
    report_to="none",
    fp16=torch.cuda.is_available()  # Mixed precision is recommended
)

# ----------------------------------------------------------------
# 6. Start training
# ----------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collate_fn,
)

print("Starting training...")
trainer.train()

# Save final model
model.save_pretrained("./siglip_output/final_model")
processor.save_pretrained("./siglip_output/final_model")
print("Training complete.")

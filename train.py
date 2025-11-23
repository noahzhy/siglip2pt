import torch
import torch.nn as nn
from PIL import Image
from transformers import SiglipProcessor, SiglipModel, Trainer, TrainingArguments
from datasets import Dataset

# ----------------------------------------------------------------
# 1. Define model wrapper (critical step)
# ----------------------------------------------------------------
class SiglipForFineTuning(nn.Module):
    """
    Wrap HuggingFace's SiglipModel so we can compute the Loss in forward().
    """
    def __init__(self, model_id):
        super().__init__()
        self.model = SiglipModel.from_pretrained(model_id)
        # The core of SigLIP uses sigmoid-based loss instead of softmax.
        # We use BCEWithLogitsLoss where the target matrix is an identity
        # matrix (1.0 on the diagonal, 0.0 elsewhere).
        self.loss_fct = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, pixel_values, attention_mask=None, **kwargs):
        # 1. Get model outputs
        outputs = self.model(
            input_ids=input_ids, 
            pixel_values=pixel_values, 
            attention_mask=attention_mask
        )

        # `logits_per_image` has shape (batch_size, batch_size)
        # It already includes SigLIP-specific bias and temperature scaling.
        logits = outputs.logits_per_image
        
        # 2. Build labels
        # Diagonal entries are positive samples (1.0), others are negatives (0.0)
        batch_size = logits.shape[0]
        labels = torch.eye(batch_size, device=logits.device)
        
        # 3. Compute loss
        loss = self.loss_fct(logits, labels)
        
        # 4. Must return a dict containing the 'loss' key for Trainer to work
        return {"loss": loss, "logits": logits}

    # Allow the Trainer to save the internal HF model
    def save_pretrained(self, save_directory):
        self.model.save_pretrained(save_directory)

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
        img = Image.new('RGB', (384, 384), color=(i*5 % 255, (i*10)%255, 150))
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
    per_device_train_batch_size=4,  # The larger the better if VRAM allows
    num_train_epochs=3,
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
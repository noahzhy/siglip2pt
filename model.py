import torch
import torch.nn as nn
from transformers import SiglipProcessor, SiglipModel


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

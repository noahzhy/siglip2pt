# SigLip2 Post-Training

This repository contains code for fine-tuning the SigLip2 model on custom datasets. SigLip2 is a state-of-the-art model for image-text tasks, and this code allows you to adapt it to your specific needs.

## Requirements

Recommended to use `uv` to manage virtual environments.

```bash
uv venv .venv
uv .venv/bin/activate
uv pip install -r requirements.txt
```

If you lived in China, you may need to use a mirror for huggingface downloads:

```bash
bash fuckGFW.sh
```

## Usage

### Training
To train the SigLip2 model on your dataset, run the following command:

```bash
python train.py
```

Make sure to adjust the configuration parameters in `config.yaml` as needed for your training setup.

### Evaluation

### Exporting to ONNX
To export the trained model to ONNX format for inference, use the following command:
```bash
python utils/onnx_export_simplify.py    \
    --model_path ./siglip_output        \
    --output_path ./siglip2.onnx        \
    --sim_path ./siglip2_sim.onnx       \
    --batch 1
```

## Speed Testing
To test the inference speed of the exported ONNX model, run:
NOTICE: threads > 1 with CUDA Graph may not yield correct results due to shared graph capture.

```bash
python utils/onnx_speed_test.py         \
    --model siglip_vision_sim.onnx      \
    --provider cuda                     \
    --use_io_binding                    \
    --enable_cuda_graph
```

## Configuration
The training parameters can be adjusted in the `config.yaml` file. Key parameters include:
- `model_id`: The pre-trained SigLip2 model to use.
- `image_input_shape`: The shape of the input images.
- `text_input_length`: The maximum length of input text.
- `batch_size`: The batch size for training.
- `learning_rate`: The learning rate for the optimizer.
- `num_epochs`: The number of training epochs.
- `train_output_dir`: The directory to save the trained model outputs.

# LocalDiT - Local Attention Diffusion Transformer

LocalDiT is an efficient implementation of Diffusion Transformers (DiT) that uses window-based local attention mechanisms to reduce computational complexity while maintaining high-quality image generation capabilities.

![LocalDiT Architecture](https://github.com/user-attachments/assets/d5e49e71-163a-4280-8f26-6fe424e9592d)

## Features

- **Window-based Local Attention**: Efficient attention computation using shifted windows similar to Swin Transformer
- **RoPE (Rotary Position Embeddings)**: Better position encoding for improved spatial understanding  
- **Flexible Architecture**: Support for various model sizes and configurations
- **Multiple Generation Modes**: Both text-to-image and image-to-image generation
- **Memory Efficient**: Reduced memory footprint compared to full attention transformers

## Model Description

LocalDiT builds upon the architecture of [PixArt-α](https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS), introducing local attention mechanisms to reduce computational complexity and memory requirements. By processing image patches in local windows rather than with global attention, the model achieves faster inference and training while preserving image generation quality.

- **Type**: Diffusion Transformer (DiT) with Local Attention
- **Parameters**: 0.52B
- **Resolution**: Supports generation up to 1024×1024 pixels (default: 768×1024)
- **Language Support**: English text prompts
- **Text Encoder**: FLAN-T5-XXL (4.3B parameters)
- **VAE**: SDXL VAE for high-quality latent encoding/decoding

## Installation

### Requirements

- Python 3.8 or higher
- PyTorch 2.0 or higher  
- CUDA-capable GPU (recommended)

### Dependencies

The package will automatically install the following dependencies:
- torch>=2.0.0
- accelerate>=0.20.0
- diffusers>=0.25.0
- transformers>=4.30.0
- einops>=0.6.0
- safetensors>=0.4.0
- pillow>=9.5.0
- numpy>=1.24.0
- tqdm>=4.65.0

## Model Weights

To download the model file, use wget to install it from the following path:
```bash
wget https://huggingface.co/DATAGRID-research/DATAGRID-Local-Attention-DiT-v1.0.0-0.52B/resolve/main/LocalDiT-1024.pt -P output/
```

## Quick Start

### Text-to-Image Generation

```python
from localdit.inference import inference
from localdit.utils import load_model
from config import Config

# Initialize configuration
config = Config()

# Load model
pipe = load_model(config)

# Generate images
prompts = ["A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece"]
neg_prompts = ["bad quality, worst quality, low resolution, blurry"]

images = inference(pipe, prompts, neg_prompts, config)

# Save images
for i, img in enumerate(images):
    img.save(f"output/generated_{i}.png")
```

### Image-to-Image Generation

```python
from PIL import Image
from localdit.inference import inference
from localdit.utils import load_model
from config import Config

# Load input image
init_image = Image.open("input_image.jpg").convert("RGB")

# Initialize and load model
config = Config()
pipe = load_model(config)

# Transform image
images = inference(pipe, prompts, neg_prompts, config, image=init_image)

# Save result
images[0].save("output/transformed.png")
```

## Running Inference Scripts

### Text-to-Image

```bash
python infer_t2i.py
```

This will generate images using predefined prompts and save them to the `output/` directory.

### Image-to-Image

```bash
python infer_i2i.py
```

This will transform an existing image using the model. By default, it uses the first generated image from text-to-image as input.

## Model Architecture

LocalDiT uses a transformer-based architecture with the following key components:

1. **Patch Embedding**: Converts image patches into embeddings
2. **Transformer Blocks**: 18 layers with:
   - Window-based self-attention with shift mechanism (window size: 4)
   - Cross-attention for text conditioning
   - Feed-forward networks
   - RMSNorm for stable training
3. **Local Attention Mechanism**:
   - Divides feature maps into non-overlapping windows
   - Computes attention within each window
   - Uses shifted windows in alternating layers for cross-window connections
   - Incorporates RoPE for better position understanding
   - Optional global context attention
4. **Output Projection**: Converts embeddings back to image space

## Model Checkpoints

Model checkpoints can be in either SafeTensors format (`.safetensors`) or PyTorch format (`.pt`). The checkpoint contains the transformer state dict with:

- Transformer blocks weights
- Attention processor states  
- Normalization parameters
- Position encoding weights

## Limitations

1. The model primarily works with English text prompts
2. Like other text-to-image models, it may struggle with complex spatial relationships, text rendering, and accurate human/animal anatomy
3. The model may inherit biases present in the training data
4. Local attention windows may occasionally miss long-range dependencies

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.
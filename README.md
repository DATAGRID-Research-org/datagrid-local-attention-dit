![スクリーンショット 2025-05-17 22 06 25](https://github.com/user-attachments/assets/d5e49e71-163a-4280-8f26-6fe424e9592d)

# LocalDiT
LocalDiT is a lightweight Diffusion Transformer model for high-quality text-to-image generation that incorporates local attention mechanisms to improve computational efficiency while maintaining generation quality.

# Model Description
LocalDiT builds upon the architecture of [PixArt-α](https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS), introducing local attention mechanisms to reduce computational complexity and memory requirements. By processing image patches in local windows rather than with global attention, the model achieves faster inference and training while preserving image generation quality.

- **Type**: Diffusion Transformer (DiT) with Local Attention
- **Parameters**: 0.52B
- **Resolution**: Supports generation up to 1024×1024 pixels
- **Language Support**: English text prompts
- **Text Encoder**: FLAN-T5-XXL (4.3B parameters)
- **VAE**: SDXL VAE for high-quality latent encoding/decoding

# Usage
Details on code execution will be released at a later date.
```python
from model import LocalDiTPipeline
import torch

pipe = LocalDiTPipeline.from_pretrained("datagrid/LocalDiT-1024", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "A cute cat sitting on a windowsill, digital art"
negative_prompt = "low quality, distorted, blurry"

image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=50).images[0]
image.save("generated_image.png")
```

# Training Details

- **Training Data**: Approximately 40M image-text pairs
- **Training Strategy**: Multi-stage resolution training (256px → 512px → 1024px)
- **Architecture Modifications**:
   - Implemented window-based local attention in alternating transformer blocks
   - Reduced parameter count through efficient attention design
   - Optimized for both quality and computational efficiency
- **Components**:
   - Diffusion Backbone: Custom LocalDiT architecture (0.52B parameters)
   - Text Encoder: FLAN-T5-XXL (4.3B parameters) for rich text embedding
   - VAE: SDXL's Variational Autoencoder for high-fidelity latent space encoding/decoding

# Results
LocalDiT achieves comparable image quality to PixArt-α while offering:
- 20% reduction in model parameters
- Up to 30% faster inference speed
- Reduced memory footprint

# License
This model is released under the Apache 2.0 License.

# Limitations
1. The model primarily works with English text prompts
2. Like other text-to-image models, it struggle with complex spatial relationships, text rendering, and accurate human/animal anatomy
3. The model may inherit biases present in the training data

# Citation
Citation information will be provided at a later date.

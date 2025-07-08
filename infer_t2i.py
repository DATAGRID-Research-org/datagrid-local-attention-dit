from localdit.inference import inference
from localdit.utils import load_model
from config import Config

config = Config()

pipe = load_model(config)
    
# Example prompts
prompts = ["A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece"] * config.batch_size
neg_prompts = ["bad quality, worst quality, low resolution, blurry, pixelated, text, frame, cartoon"] * config.batch_size

# Generate images
images = inference(pipe, prompts, neg_prompts, config)

# Save images
for i, img in enumerate(images):
    img.save(f"output/generated_image_{i}.png")
import os
import copy

from PIL import Image
import torch
from safetensors.torch import load_file
from transformers import T5EncoderModel, T5Tokenizer
from diffusers import (
    PixArtTransformer2DModel, 
    DPMSolverMultistepScheduler,
    AutoencoderKL,
    PixArtSigmaPipeline
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.normalization import RMSNorm

from localdit.models.attentions import GCAttnProcessor2_0
from localdit.models.ema import EMA

def load_model(config, device=None, dtype=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = PixArtTransformer2DModel(
        num_attention_heads=config.num_attention_heads,
        attention_head_dim=config.attention_head_dim,
        num_embeds_ada_norm=config.num_embeds_ada_norm,
        cross_attention_dim=config.cross_attention_dim,
        caption_channels=config.caption_channels,
        patch_size=config.patch_size,
        sample_size=config.width//8,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        num_layers=config.num_layers,
        norm_type=config.norm_type,
        use_additional_conditions=True,
        dropout=config.dropout,).to(device=device, dtype=dtype)
    
    for i in range(config.num_layers):
        model.transformer_blocks[i].attn1.fuse_projections()
        model.transformer_blocks[i].attn1.norm_q = RMSNorm(
            config.attention_head_dim, eps=config.eps)
        model.transformer_blocks[i].attn1.norm_k = RMSNorm(
            config.attention_head_dim, eps=config.eps)
        model.transformer_blocks[i].attn2.fuse_projections()
        model.transformer_blocks[i].attn2.norm_q = RMSNorm(
            config.attention_head_dim, eps=config.eps)
        model.transformer_blocks[i].attn2.norm_k = RMSNorm(
            config.attention_head_dim, eps=config.eps)

    for i in range(config.num_layers):
        shift_size = i % config.window_size
        model.transformer_blocks[i].attn1.set_processor(
            GCAttnProcessor2_0(
                window_size=config.window_size,  # Adjust as needed
                num_heads=config.num_attention_heads,
                use_global=True,
                shift_size=shift_size,
                dim=config.cross_attention_dim,
                do_rope=config.do_rope,
                device=device,
                dtype=dtype
            )
        )

    # ema = EMA(beta=config.ema_beta)
    # ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    if config.ckpt_path.endswith(".safetensors"):
        model_state_dict = load_file(config.ckpt_path)
    else:
        model_state_dict = torch.load(config.ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(model_state_dict, strict=False)
    # ema_model_state_dict = torch.load(config.ckpt_ema_path, map_location=DEVICE, weights_only=False)
    # ema_model.load_state_dict(ema_model_state_dict, strict=False)

    # Load model components
    vae = AutoencoderKL.from_pretrained(
        "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers", 
        subfolder="vae", 
        torch_dtype=dtype).to(device=device)
    
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
    text_encoder = T5EncoderModel.from_pretrained(
        "google/flan-t5-xxl", 
        torch_dtype=dtype).to(device=device)
    
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers", 
        subfolder="scheduler", 
        torch_dtype=dtype)

    # Create pipeline
    pipe = PixArtSigmaPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        transformer=model,
        scheduler=scheduler,
    )
    pipe.to(device=device, dtype=dtype)
    
    return pipe

def prepare_image_latents(pipe, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
    def retrieve_latents(
        encoder_output: torch.Tensor, generator = None, sample_mode: str = "sample"
    ):
        if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
            return encoder_output.latent_dist.sample(generator)
        elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
            return encoder_output.latent_dist.mode()
        elif hasattr(encoder_output, "latents"):
            return encoder_output.latents
        else:
            raise AttributeError("Could not access latents of provided encoder_output")
    
    if isinstance(image, Image.Image):
        image = pipe.image_processor.preprocess(image)

    image = image.to(device=device, dtype=dtype)

    batch_size = batch_size * num_images_per_prompt

    if image.shape[1] == 4:
        init_latents = image

    else:
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        elif isinstance(generator, list):
            if image.shape[0] < batch_size and batch_size % image.shape[0] == 0:
                image = torch.cat([image] * (batch_size // image.shape[0]), dim=0)
            elif image.shape[0] < batch_size and batch_size % image.shape[0] != 0:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {image.shape[0]} to effective batch_size {batch_size} "
                )

            init_latents = [
                retrieve_latents(pipe.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(batch_size)
            ]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = retrieve_latents(pipe.vae.encode(image), generator=generator)

        init_latents = pipe.vae.config.scaling_factor * init_latents

    init_latents = torch.cat([init_latents], dim=0)

    shape = init_latents.shape
    noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    # get latents
    init_latents = pipe.scheduler.add_noise(init_latents, noise, timestep)
    latents = init_latents
    
    return latents
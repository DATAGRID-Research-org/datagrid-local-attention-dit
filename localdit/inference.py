import torch
import tqdm
from diffusers import PixArtSigmaPipeline

from localdit.utils import prepare_image_latents

@torch.no_grad()
def inference(pipe: PixArtSigmaPipeline, prompts, neg_prompts, config, image=None):
    device = pipe.device
    dtype = pipe.dtype
    width_vae = config.width // 16
    height_vae = config.height // 16
    for i in range(config.num_layers):
        # Pass integer height and width
        pipe.transformer.transformer_blocks[i].attn1.processor.set_image_size(height_vae, width_vae)

    batch_size = len(prompts)
    prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = encode_prompt(
        pipe, prompt=prompts, negative_prompt=neg_prompts, max_sequence_length=config.max_length, device=device)
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
    
    pipe.scheduler.set_timesteps(config.denoising_steps, device=device)
    if image is None:
        timesteps = pipe.scheduler.timesteps
        latents = pipe.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=config.in_channels,
            height=config.height,
            width=config.width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=None,
            latents=None)
    else:
        init_timestep = min(int(config.denoising_steps * config.denoising_strength), config.denoising_steps)
        t_start = max(config.denoising_steps - init_timestep, 0)
        timesteps = pipe.scheduler.timesteps[t_start * pipe.scheduler.order:]
        latents = prepare_image_latents(
            pipe, image, timesteps[:1].repeat(batch_size),
            batch_size=batch_size,
            num_images_per_prompt=1,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=None)

    if neg_prompts is not None:
        batch_size *= 2
    
    resolution_tensor = torch.tensor([[config.height, config.width] for _ in range(batch_size)]).to(device=device, dtype=dtype)
    aspect_ratio_tensor = torch.tensor([config.width / config.height for _ in range(batch_size)]).to(device=device, dtype=dtype)

    pbar = tqdm.tqdm(total=len(timesteps), desc="Generating images")
    for t in timesteps:
        latent_model_input = torch.cat([latents] * 2) if neg_prompts is not None else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        current_timestep = t
        current_timestep = current_timestep.expand(latent_model_input.shape[0])
        noise_pred = pipe.transformer(
            latent_model_input,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            timestep=current_timestep,
            added_cond_kwargs={"resolution": resolution_tensor, "aspect_ratio": aspect_ratio_tensor},
            return_dict=False)[0]
            
        if neg_prompts is not None:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + config.guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        pbar.update(1)
    pbar.close()
    images = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    images = pipe.image_processor.postprocess(images, output_type="pil")
    return images

def encode_prompt(pipe, prompt, negative_prompt, max_sequence_length, device):
    if isinstance(prompt, str):
        batch_size = 1
    else:
        batch_size = len(prompt)

    # Encode positive prompt
    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    prompt_attention_mask = text_inputs.attention_mask.to(device)
    prompt_embeds = pipe.text_encoder(text_inputs.input_ids.to(device), attention_mask=prompt_attention_mask)[0]

    # Encode negative prompt
    uncond_input = pipe.tokenizer(
        negative_prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    negative_prompt_attention_mask = uncond_input.attention_mask.to(device)
    negative_prompt_embeds = pipe.text_encoder(
        uncond_input.input_ids.to(device), attention_mask=negative_prompt_attention_mask)[0]

    return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask
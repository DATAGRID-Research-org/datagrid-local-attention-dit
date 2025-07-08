from dataclasses import dataclass
import glob
import copy
import datetime
import random
import argparse
import os
import io
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
WORLD_RANK = int(os.getenv("RANK", "0"))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))

import numpy as np
import deepspeed as ds
import safetensors
import torch
import tqdm
import torch.distributed as dist
dist.init_process_group(
    backend="nccl",
    world_size=WORLD_SIZE,
    rank=WORLD_RANK)
torch.cuda.set_device(LOCAL_RANK)
torch.set_float32_matmul_precision('high')
DEVICE = torch.device("cuda", LOCAL_RANK)
DTYPE = torch.bfloat16

# set random seed
current_time_seed_base = int(datetime.datetime.now().timestamp())
rank_offset = WORLD_RANK * 20 + LOCAL_RANK
seed = current_time_seed_base + rank_offset
seed = abs(seed) # Ensure non-negative
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed) # Seeds all GPUs

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

if True:
    import wandb

import accelerate
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler
from diffusers import PixArtSigmaPipeline
from diffusers.models import AutoencoderKL
from diffusers.models.normalization import RMSNorm
from diffusers.models.transformers import PixArtTransformer2DModel
from diffusers.models.attention_processor import Attention, FusedAttnProcessor2_0
from transformers import T5EncoderModel, T5Tokenizer

from localdit.models.attentions import GCAttnProcessor2_0
from prompt import val_prompts

validation_prompts = val_prompts[0]# + val_prompts[1] + val_prompts[2] + val_prompts[3] + val_prompts[4] + val_prompts[5]

validation_neg_prompts = [
    "bad quality, worst quality, low resolution, blurry, pixelated, text, frame, cartoon" for _ in range(len(validation_prompts))
]

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

@torch.no_grad()
def validation(pipe:PixArtSigmaPipeline, prompts, neg_prompts, config):
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(123456789+WORLD_RANK)
    
    batch_size = len(prompts)
    prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = encode_prompt(
        pipe, prompt=prompts, negative_prompt=neg_prompts, max_sequence_length=config.max_length, device=DEVICE)
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
    
    pipe.scheduler.set_timesteps(config.validation_steps, device=DEVICE)
    timesteps = pipe.scheduler.timesteps
    latents = pipe.prepare_latents(
        batch_size=batch_size,
        num_channels_latents=config.in_channels,
        height=config.image_size,
        width=config.image_size,
        dtype=prompt_embeds.dtype,
        generator=generator,
        device=DEVICE,
        latents=None)
    if accelerator.is_local_main_process:
        pbar = tqdm.tqdm(timesteps, desc="Validation")
    for i, t in enumerate(timesteps):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        current_timestep = t
        current_timestep = current_timestep.expand(latent_model_input.shape[0])
        noise_pred = pipe.transformer(
            latent_model_input,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            timestep=current_timestep,
            added_cond_kwargs={
                "resolution": torch.tensor([[config.image_size, config.image_size] for _ in range(batch_size*2)]).to(device=DEVICE, dtype=DTYPE),
                "aspect_ratio": torch.tensor([1.0 for _ in range(batch_size*2)]).to(device=DEVICE, dtype=DTYPE)},
            return_dict=False,)[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + config.guidance_scale * (noise_pred_text - noise_pred_uncond)
        # noise_pred = noise_pred.chunk(2, dim=1)[0]
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        if accelerator.is_local_main_process:
            pbar.update(1)
    if accelerator.is_local_main_process:
        pbar.close()
        
    val_images = pipe.vae.decode(latents / scaling_factor, return_dict=False)[0]
    val_images = pipe.image_processor.postprocess(val_images, output_type="pil")
    return val_images

def to_tensor(data, dtype=None, device=None):
    return torch.from_dlpack(data.as_tensor()).to(device=device, dtype=dtype)

def decode_caption(data):
    dropout = np.random.rand(1)
    if dropout < 0.1:
        return ""
    return np.array(data).tobytes().decode('utf-8')

def prepare_batch(batch, step:int=0):
    img, cap = batch
    img = to_tensor(img)
    cap = list(map(decode_caption, cap))
    return img, cap

def encode_prompt(
    pipe,
    prompt="",
    token=None,
    token_mask=None,
    do_classifier_free_guidance: bool = True,
    negative_prompt: str = "",
    num_images_per_prompt: int = 1,
    device = None,
    prompt_embeds = None,
    negative_prompt_embeds = None,
    prompt_attention_mask = None,
    negative_prompt_attention_mask = None,
    clean_caption: bool = False,
    max_sequence_length: int = 300,
    **kwargs):
    if device is None:
        device = pipe._execution_device

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    # See Section 3.1. of the paper.
    max_length = max_sequence_length

    if prompt_embeds is None:
        if token is None:
            prompt = pipe._text_preprocessing(prompt, clean_caption=clean_caption)
            text_inputs = pipe.tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            
            prompt_attention_mask = text_inputs.attention_mask
            prompt_attention_mask = prompt_attention_mask.to(device)
        else:
            text_input_ids = token.to(device)
            prompt_attention_mask = token_mask.to(device)

        prompt_embeds = pipe.text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask)
        prompt_embeds = prompt_embeds[0]

    if pipe.text_encoder is not None:
        dtype = pipe.text_encoder.dtype
    elif pipe.transformer is not None:
        dtype = pipe.transformer.dtype
    else:
        dtype = None

    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    bs_embed, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
    prompt_attention_mask = prompt_attention_mask.view(bs_embed, -1)
    prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)

    # get unconditional embeddings for classifier free guidance
    if do_classifier_free_guidance and negative_prompt_embeds is None:
        uncond_tokens = [negative_prompt] * batch_size if isinstance(negative_prompt, str) else negative_prompt
        uncond_tokens = pipe._text_preprocessing(uncond_tokens, clean_caption=clean_caption)
        max_length = prompt_embeds.shape[1]
        uncond_input = pipe.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        negative_prompt_attention_mask = uncond_input.attention_mask
        negative_prompt_attention_mask = negative_prompt_attention_mask.to(device)

        negative_prompt_embeds = pipe.text_encoder(
            uncond_input.input_ids.to(device), attention_mask=negative_prompt_attention_mask
        )
        negative_prompt_embeds = negative_prompt_embeds[0]

    if do_classifier_free_guidance:
        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = negative_prompt_embeds.shape[1]

        negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)

        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        negative_prompt_attention_mask = negative_prompt_attention_mask.view(bs_embed, -1)
        negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(num_images_per_prompt, 1)
    else:
        negative_prompt_embeds = None
        negative_prompt_attention_mask = None

    return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

def init_weight(cls): # inspired by miyake-san
    def _basic_init(module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)

    cls.apply(_basic_init)

    w = cls.pos_embed.proj.weight.data
    torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
    torch.nn.init.normal_(cls.adaln_single.emb.timestep_embedder.linear_1.weight, std=0.02)
    torch.nn.init.normal_(cls.adaln_single.emb.timestep_embedder.linear_2.weight, std=0.02)
    torch.nn.init.normal_(cls.adaln_single.linear.weight, std=0.02)
    torch.nn.init.normal_(cls.caption_projection.linear_1.weight, std=0.02)
    torch.nn.init.normal_(cls.caption_projection.linear_2.weight, std=0.02)
    for block in cls.transformer_blocks:
        torch.nn.init.constant_(block.attn2.to_out[0].weight, 0)
        torch.nn.init.constant_(block.attn2.to_out[0].bias, 0)
    torch.nn.init.constant_(cls.proj_out.weight, 0)
    torch.nn.init.constant_(cls.proj_out.bias, 0)

@dataclass
class TrainingConfig:
    num_workers = 32
    attention_head_dim = 72
    validation_steps = 20
    train_sampling_steps = 1000
    save_checkpoint_interval = 5_000
    validation_interval = 1_000
    image_size = 1024
    batch_size = 12
    learning_rate = 2e-5
    weight_decay = 3e-2
    eps = 1e-10
    gradient_clip = 0.01
    max_length = 120
    max_datasteps = 400_000
    guidance_scale = 7.0
    in_channels = 4
    out_channels = 4
    num_attention_heads = 16
    patch_size = 2
    window_size = 4 # 4, 8, 16
    num_layers = 18
    norm_type = "ada_norm_single"
    cross_attention_dim = 1152
    caption_channels = 4096
    num_embeds_ada_norm = 1000
    token_dropout = 0.10
    dropout = 0
    ema_beta = 0.9
    ema_step = 350_000
config = TrainingConfig()

accelerator = accelerate.Accelerator(
    deepspeed_plugin=accelerate.DeepSpeedPlugin(
        hf_ds_config={
            "bf16": {
                "enabled": True,
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "eps": config.eps,
                    "torch_adam": True,
                    "adam_w_mode": True,
                }
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": config.learning_rate,
                    "warmup_num_steps": 1000,
                }
            },
            "zero_optimization": {
                "stage": 2,
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "contiguous_gradients": True
            },
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": config.batch_size,
        }))

data_paths = []
# this is where you put the path to the data
data_paths.sort()

index_paths = []
# for webdataset index files
index_paths.sort()

@pipeline_def(
    batch_size=config.batch_size,
    num_threads=config.num_workers,
    device_id=LOCAL_RANK)
def wds_pipeline(wds_data=data_paths, wds_index=index_paths):
    ext = ["jpg", "cap"]
    img_raw, cap = fn.readers.webdataset(
        paths=wds_data,
        index_paths=wds_index,
        seed=seed,
        ext=ext, missing_component_behavior="error", random_shuffle=True
    )
    img = fn.decoders.image(img_raw, device="mixed", output_type=types.RGB)
    img = 2 * fn.crop_mirror_normalize(img, dtype=types.FLOAT,)/255.0 - 1
    return img, cap

dataloader= wds_pipeline()
dataloader.build()

model = PixArtTransformer2DModel(
    num_attention_heads=config.num_attention_heads,
    attention_head_dim=config.attention_head_dim,
    num_embeds_ada_norm=config.num_embeds_ada_norm,
    cross_attention_dim=config.cross_attention_dim,
    caption_channels=config.caption_channels,
    patch_size=config.patch_size,
    sample_size=config.image_size//8,
    in_channels=config.in_channels,
    out_channels=config.out_channels,
    num_layers=config.num_layers,
    norm_type=config.norm_type,
    use_additional_conditions=True,
    dropout=config.dropout,).to(device=DEVICE, dtype=DTYPE)

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
            device=DEVICE,
            dtype=DTYPE
        ))
    
init_weight(model)

vae = AutoencoderKL.from_pretrained(
    "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers", subfolder="vae", torch_dtype=DTYPE).to(device=DEVICE)
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
text_encoder = T5EncoderModel.from_pretrained("google/flan-t5-xxl", torch_dtype=DTYPE).to(device=DEVICE)
scheduler = DPMSolverMultistepScheduler.from_pretrained(
    "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers", subfolder="scheduler", torch_dtype=DTYPE)
noise_scheduler = DDIMScheduler.from_config(scheduler.config)

scaling_factor = vae.config.scaling_factor
pipe = PixArtSigmaPipeline(
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    vae=vae,
    transformer=model,
    scheduler=scheduler,
)
pipe.to(device=DEVICE, dtype=DTYPE)
optimizer = accelerate.utils.DummyOptim(
    model.parameters(), lr=config.learning_rate)
lr_scheduler = accelerate.utils.DummyScheduler(optimizer)

ema = EMA(beta=config.ema_beta)
ema_model = copy.deepcopy(model).eval().requires_grad_(False)

start_step = 0
resume = 320_000
resume_size = 1024
if resume > 0:
    model_state_dict = torch.load(f"./{resume_size}x{resume_size}/ckpt/LocalDiT-{resume_size}x{resume_size}-{resume}.pt", weights_only=False)
    model.load_state_dict(model_state_dict, strict=False)
    ema_model_state_dict = torch.load(f"./{resume_size}x{resume_size}/ckpt/LocalDiT-{resume_size}x{resume_size}-{resume}-ema.pt", weights_only=False)
    ema_model.load_state_dict(ema_model_state_dict, strict=False)
    start_step = resume
    print(f"Resume from {start_step} step")

model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
model.train()

if LOCAL_RANK == 0:
    run = wandb.init(
        ...
    )
    print("Start training")
    pbar = tqdm.tqdm(total=config.max_datasteps-start_step, desc=f"Global step {start_step}/{config.max_datasteps}")
    
for global_step in range(start_step, config.max_datasteps+1):           
    batch = dataloader.run()
    x, caption = prepare_batch(batch, global_step)
    
    with torch.no_grad():
        x = x.to(device=DEVICE, dtype=DTYPE)
        z = pipe.vae.encode(x).latent_dist.sample()
        clean_images = z * scaling_factor
        prompt_embeds, prompt_attention_mask, _, _ = encode_prompt(
            pipe,
            caption,
            max_sequence_length=config.max_length,
            do_classifier_free_guidance=False,
            device=DEVICE)
    
    batch_size = clean_images.shape[0]
    timesteps = torch.randint(
        0, config.train_sampling_steps, (batch_size,), device=clean_images.device).long()
    noise = torch.randn_like(clean_images)
    noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
    with accelerator.accumulate(model) and accelerator.autocast():
        noise_pred = model(
            noisy_images,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            added_cond_kwargs={
                "resolution": torch.tensor([[config.image_size, config.image_size] for _ in range(batch_size)]).to(device=DEVICE, dtype=DTYPE),
                "aspect_ratio": torch.tensor([1.0 for _ in range(batch_size)]).to(device=DEVICE, dtype=DTYPE)},
            return_dict=False,)[0]
        
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        accelerator.backward(loss)
        if LOCAL_RANK == 0:
            pbar.set_postfix({"loss": loss.item()})
            pbar.set_description(f"Global step {global_step}/{config.max_datasteps}")
            pbar.update(1)

        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if global_step > config.ema_step:
            raw_model = accelerator.unwrap_model(model)
            ema.step_ema(ema_model, raw_model, step_start_ema=1)

        if global_step % config.save_checkpoint_interval == 0:
            if not os.path.exists(f"./{config.image_size}x{config.image_size}/ckpt"):
                os.makedirs(f"./{config.image_size}x{config.image_size}/ckpt", exist_ok=True)
            ema_state_dict = ema_model.state_dict()
            torch.save(ema_state_dict, f"./{config.image_size}x{config.image_size}/ckpt/LocalDiT-{config.image_size}x{config.image_size}-{global_step}-ema.pt")
            state_dict = accelerator.get_state_dict(model)
            torch.save(state_dict, f"./{config.image_size}x{config.image_size}/ckpt/LocalDiT-{config.image_size}x{config.image_size}-{global_step}.pt")
            raw_optim = accelerator.unwrap_model(optimizer)
            torch.save(raw_optim.state_dict(), f"./{config.image_size}x{config.image_size}/ckpt/LocalDiT-{config.image_size}x{config.image_size}-{global_step}-optim-{WORLD_RANK}.pt")
            raw_sched = accelerator.unwrap_model(lr_scheduler)
            torch.save(raw_sched.state_dict(), f"./{config.image_size}x{config.image_size}/ckpt/LocalDiT-{config.image_size}x{config.image_size}-{global_step}-sched-{WORLD_RANK}.pt")
        if LOCAL_RANK == 0:
            run.log({"loss": loss.item()}, step=global_step)

            if global_step % config.validation_interval == 0:
                if global_step <= config.ema_step:
                    raw_model = accelerator.unwrap_model(model)
                    ema_model.load_state_dict(raw_model.state_dict())

                pipe.transformer = ema_model
                with torch.no_grad():
                    val_images = validation(pipe, validation_prompts, validation_neg_prompts, config)
                    run.log({"sample": [wandb.Image(image, caption=validation_prompts[img_idx]) for img_idx, image in enumerate(val_images)]}, step=global_step)
                    
        accelerator.wait_for_everyone()

accelerator.end_training()
if LOCAL_RANK == 0:
    print("Training completed")
    pbar.close()

from dataclasses import dataclass

@dataclass
class Config:
    num_workers = 32
    attention_head_dim = 72
    denoising_steps = 20
    denoising_strength = 0.8
    train_sampling_steps = 1000
    save_checkpoint_interval = 20_000
    validation_interval = 1_000
    height = 768
    width = 1024
    batch_size = 1
    learning_rate = 5e-5
    weight_decay = 3e-2
    eps = 1e-10
    gradient_clip = 0.01
    max_length = 120
    max_datasteps = 500_000
    guidance_scale = 7.0
    in_channels = 4
    out_channels = 4
    num_attention_heads = 16
    patch_size = 2
    window_size = 4
    num_layers = 18
    norm_type = "ada_norm_single"
    cross_attention_dim = 1152
    caption_channels = 4096
    num_embeds_ada_norm = 1000
    token_dropout = 0.15
    dropout = 0
    ema_beta = 0.9999
    ema_step = 450_000
    do_rope = True
    ckpt_path = "output/LocalDit-1024.safetensors"
    # ckpt_ema_path = "ckpt/512x512/LocalDit-512x512-300000-ema.pt"
import os
import warnings
from typing import List, Optional, Union, Tuple, Dict, Any, Callable
import inspect

import torch
import numpy as np
from PIL import Image

from diffusers import DiffusionPipeline, PixArtTransformer2DModel, DPMSolverMultistepScheduler
from diffusers import AutoencoderKL, PixArtSigmaPipeline
from diffusers.models.normalization import RMSNorm
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.utils import deprecate, logging, PIL_INTERPOLATION, is_torch_xla_available
from diffusers.models import ImageProjection
from diffusers.loaders import TextualInversionLoaderMixin, IPAdapterMixin, StableDiffusionLoraLoaderMixin, FromSingleFileMixin
from transformers import T5EncoderModel, T5Tokenizer

from localdit.models.attentions import GCAttnProcessor2_0

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)

class LocalDiTTxt2ImgPipeline(DiffusionPipeline, TextualInversionLoaderMixin, IPAdapterMixin, StableDiffusionLoraLoaderMixin, FromSingleFileMixin):
    """
    Pipeline for text-to-image generation using LocalDiT.
    
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).
    """
    
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: T5EncoderModel,
        tokenizer: T5Tokenizer,
        transformer: PixArtTransformer2DModel,
        scheduler: DPMSolverMultistepScheduler,
        window_size: int = 4,
        eps: float = 1e-10
    ):
        super().__init__()
        
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )
        
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.window_size = window_size
        self.eps = eps
        
    @classmethod
    def from_pretrained(cls, pretrained_model_path=None, **kwargs):
        """
        Load a LocalDiT pipeline from a pretrained model.
        """
        # Load components
        vae = AutoencoderKL.from_pretrained(
            "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers", 
            subfolder="vae", 
            torch_dtype=torch.bfloat16)
        
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
        text_encoder = T5EncoderModel.from_pretrained(
            "google/flan-t5-xxl", 
            torch_dtype=torch.bfloat16)
            
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers", 
            subfolder="scheduler", 
            torch_dtype=torch.bfloat16)
            
        # Get transformer configuration from kwargs
        num_attention_heads = kwargs.pop("num_attention_heads", 16)
        attention_head_dim = kwargs.pop("attention_head_dim", 72)
        num_embeds_ada_norm = kwargs.pop("num_embeds_ada_norm", 1000)
        cross_attention_dim = kwargs.pop("cross_attention_dim", 1152)
        caption_channels = kwargs.pop("caption_channels", 4096)
        patch_size = kwargs.pop("patch_size", 2)
        sample_size = kwargs.pop("sample_size", 128)  # 1024 // 8
        in_channels = kwargs.pop("in_channels", 4)
        out_channels = kwargs.pop("out_channels", 4)
        num_layers = kwargs.pop("num_layers", 18)
        norm_type = kwargs.pop("norm_type", "ada_norm_single")
        do_rope = kwargs.pop("do_rope", True)
        window_size = kwargs.pop("window_size", 4)
        eps = kwargs.pop("eps", 1e-10)
            
        # Create and configure transformer
        transformer = PixArtTransformer2DModel(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_embeds_ada_norm=num_embeds_ada_norm,
            cross_attention_dim=cross_attention_dim,
            caption_channels=caption_channels,
            patch_size=patch_size,
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            norm_type=norm_type,
            use_additional_conditions=True,
            dropout=0,
        )
        
        # Configure the transformer blocks
        for i in range(num_layers):
            transformer.transformer_blocks[i].attn1.fuse_projections()
            transformer.transformer_blocks[i].attn1.norm_q = RMSNorm(
                attention_head_dim, eps=eps)
            transformer.transformer_blocks[i].attn1.norm_k = RMSNorm(
                attention_head_dim, eps=eps)
            transformer.transformer_blocks[i].attn2.fuse_projections()
            transformer.transformer_blocks[i].attn2.norm_q = RMSNorm(
                attention_head_dim, eps=eps)
            transformer.transformer_blocks[i].attn2.norm_k = RMSNorm(
                attention_head_dim, eps=eps)

        for i in range(num_layers):
            shift_size = i % window_size
            transformer.transformer_blocks[i].attn1.set_processor(
                GCAttnProcessor2_0(
                    window_size=window_size,  
                    num_heads=num_attention_heads,
                    use_global=True,
                    shift_size=shift_size,
                    dim=cross_attention_dim,
                    do_rope=do_rope,
                    device="cpu",  # Will set device when moved
                    dtype=torch.bfloat16
                )
            )
        
        # Load model weights if provided
        if pretrained_model_path is not None:
            model_state_dict = torch.load(pretrained_model_path, map_location="cpu", weights_only=False)
            transformer.load_state_dict(model_state_dict, strict=False)
        
        # Create the pipeline instance
        pipeline = cls(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
            window_size=window_size,
            eps=eps,
            **kwargs,
        )
        
        return pipeline
    
    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: Optional[int] = 120,
    ):
        """
        Encodes the prompt into text encoder hidden states.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
            
        if prompt_embeds is None:
            # Tokenize prompts
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            
            prompt_attention_mask = text_inputs.attention_mask.to(device)
            prompt_embeds = self.text_encoder(
                text_inputs.input_ids.to(device), 
                attention_mask=prompt_attention_mask
            )[0]
            
        prompt_embeds = prompt_embeds.to(device=device)
        
        # Duplicate text embeddings for each generation per prompt
        if num_images_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            
        # Get unconditional embeddings for classifier-free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_prompt = negative_prompt if negative_prompt is not None else [""] * batch_size
            
            uncond_input = self.tokenizer(
                uncond_prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            
            negative_prompt_attention_mask = uncond_input.attention_mask.to(device)
            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device), 
                attention_mask=negative_prompt_attention_mask
            )[0]
            
        if do_classifier_free_guidance:
            # Duplicate negative embeddings for each generation per prompt
            if num_images_per_prompt > 1:
                negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
                
        return prompt_embeds, negative_prompt_embeds
    
    def prepare_latents(
        self, 
        batch_size, 
        num_channels_latents, 
        height, 
        width, 
        dtype, 
        device, 
        generator=None, 
        latents=None
    ):
        """
        Prepare latents for diffusion process.
        """
        vae_scale_factor = self.vae_scale_factor
        height_vae = height // vae_scale_factor
        width_vae = width // vae_scale_factor
        shape = (batch_size, num_channels_latents, height_vae, width_vae)
        
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)
            
        # Scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        
        return latents
        
    def set_image_sizes(self, height, width):
        """
        Set image sizes for the attention processors in the transformer.
        """
        height_vae = height // self.vae_scale_factor
        width_vae = width // self.vae_scale_factor
        
        for i in range(len(self.transformer.transformer_blocks)):
            self.transformer.transformer_blocks[i].attn1.processor.set_image_size(height_vae, width_vae)
            
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = 576,
        width: Optional[int] = 1024,
        target_height: Optional[int] = 1152,
        target_width: Optional[int] = 2048,
        num_inference_steps: int = 20,
        refinement_steps: int = 20,
        refinement_strength: float = 0.5,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        max_sequence_length: int = 120,
        save_initial_image: bool = False,
        initial_output_dir: Optional[str] = "output",
    ):
        """
        Generate images from text prompt using LocalDiT.
        
        Args:
            prompt (`str` or `List[str]`): Text prompt for image generation.
            height (`int`, *optional*, defaults to 576): Initial height of the generated image.
            width (`int`, *optional*, defaults to 1024): Initial width of the generated image.
            target_height (`int`, *optional*, defaults to 1152): Target height for image refinement.
            target_width (`int`, *optional*, defaults to 2048): Target width for image refinement.
            num_inference_steps (`int`, *optional*, defaults to 20): Number of inference steps for initial image generation.
            refinement_steps (`int`, *optional*, defaults to 20): Number of refinement steps.
            refinement_strength (`float`, *optional*, defaults to 0.5): Strength of refinement (0-1).
            guidance_scale (`float`, *optional*, defaults to 7.0): Guidance scale for classifier-free guidance.
            negative_prompt (`str` or `List[str]`, *optional*): Negative text prompt for classifier-free guidance.
            num_images_per_prompt (`int`, *optional*, defaults to 1): Number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0): Corresponds to parameter eta (η) from DDIM paper.
            generator (`torch.Generator`, *optional*): A torch generator for reproducibility.
            latents (`torch.Tensor`, *optional*): Pre-generated noise tensor.
            prompt_embeds (`torch.Tensor`, *optional*): Pre-computed text embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*): Pre-computed negative text embeddings.
            output_type (`str`, *optional*, defaults to "pil"): Output format ("pil", "latent" or "np").
            return_dict (`bool`, *optional*, defaults to True): Whether to return a dictionary or tuple.
            callback (`Callable`, *optional*): Callback function on each step.
            callback_steps (`int`, *optional*, defaults to 1): Frequency of callback function.
            max_sequence_length (`int`, *optional*, defaults to 120): Maximum text length for tokenization.
            save_initial_image (`bool`, *optional*, defaults to False): Whether to save the initial generated image.
            initial_output_dir (`str`, *optional*, defaults to "output"): Output directory for initial images.
        
        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: An object containing the generated images.
        """
        # 0. Default values and type conversion
        if height % self.vae_scale_factor != 0 or width % self.vae_scale_factor != 0:
            raise ValueError(f"height and width must be divisible by {self.vae_scale_factor}")
        if target_height % self.vae_scale_factor != 0 or target_width % self.vae_scale_factor != 0:
            raise ValueError(f"target_height and target_width must be divisible by {self.vae_scale_factor}")
            
        # Check aspect ratio
        initial_ratio = width / height
        target_ratio = target_width / target_height
        
        if abs(initial_ratio - target_ratio) > 0.01:
            logger.warning(
                f"Initial ratio ({initial_ratio:.2f}) and target ratio ({target_ratio:.2f}) do not match. "
                f"This may cause unexpected results."
            )
            
        # 1. Process inputs
        device = self._execution_device
        dtype = next(self.transformer.parameters()).dtype
        
        # 2. Encode prompt
        do_classifier_free_guidance = guidance_scale > 1.0
        
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
        )
        
        # 3. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        # 4. Prepare latent variables
        batch_size = 1 if isinstance(prompt, str) or prompt is None else len(prompt)
        batch_size = batch_size * num_images_per_prompt
        
        latents = self.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=self.transformer.config.in_channels,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=generator,
            latents=latents,
        )
        
        # 5. Set initial image size
        aspect_ratio = width / height
        self.set_image_sizes(height, width)
        
        # 6. Prepare extra kwargs for the scheduler step
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        # 7. First-stage denoising (initial image generation)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        
        logger.info(f"Generating initial image: {width}x{height}")
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand the latents for classifier-free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                # Predict the noise residual
                noise_pred = self.transformer(
                    latent_model_input,
                    encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds]) if do_classifier_free_guidance else prompt_embeds,
                    timestep=t,
                    added_cond_kwargs={
                        "resolution": torch.tensor([[height, width] for _ in range(batch_size * (2 if do_classifier_free_guidance else 1))]).to(device=device, dtype=dtype),
                        "aspect_ratio": torch.tensor([aspect_ratio for _ in range(batch_size * (2 if do_classifier_free_guidance else 1))]).to(device=device, dtype=dtype)
                    },
                    return_dict=False,
                )[0]
                
                # Classifier-free guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # Compute previous noisy sample
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                
                # Call callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
                
                if XLA_AVAILABLE:
                    xm.mark_step()
        
        # 8. Decode the image
        initial_images = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        initial_images = self.image_processor.postprocess(initial_images, output_type="pil")
        
        # Optionally save initial images
        if save_initial_image:
            os.makedirs(initial_output_dir, exist_ok=True)
            for i, img in enumerate(initial_images):
                path = os.path.join(initial_output_dir, f"initial_{i}.jpg")
                img.save(path)
                logger.info(f"Initial image saved: {path}")
        
        # 9. Refinement stage
        # Resize images to target dimensions
        resized_images = []
        for img in initial_images:
            resized_img = img.resize((target_width, target_height), Image.LANCZOS)
            resized_images.append(resized_img)
        
        # Set target image size for attention processors
        self.set_image_sizes(target_height, target_width)
        
        # Strength-based timestep selection
        self.scheduler.set_timesteps(refinement_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        init_timestep = min(int(refinement_steps * refinement_strength), refinement_steps)
        t_start = max(refinement_steps - init_timestep, 0)
        timesteps = timesteps[t_start:]
        
        # First timestep
        latent_timestep = timesteps[:1].repeat(batch_size)
        
        # Prepare latents from resized image
        if isinstance(resized_images[0], Image.Image):
            image_tensors = []
            for img in resized_images:
                img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
                img_tensor = 2.0 * img_tensor - 1.0  # [-1, 1] normalization
                image_tensors.append(img_tensor)
            
            image = torch.cat(image_tensors).to(device=device, dtype=dtype)
            
            # Encode image to latent space
            latents = self.vae.encode(image).latent_dist.sample()
            latents = self.vae.config.scaling_factor * latents
            
            # Add noise based on strength
            noise = torch.randn(latents.shape, device=device, dtype=dtype)
            latents = self.scheduler.add_noise(latents, noise, latent_timestep)
        
        # Target aspect ratio
        aspect_ratio = target_width / target_height
        
        # 10. Refinement denoising process
        logger.info(f"Refining image to: {target_width}x{target_height}")
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand the latents for classifier-free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                # Predict the noise residual
                noise_pred = self.transformer(
                    latent_model_input,
                    encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds]) if do_classifier_free_guidance else prompt_embeds,
                    timestep=t,
                    added_cond_kwargs={
                        "resolution": torch.tensor([[target_height, target_width] for _ in range(batch_size * (2 if do_classifier_free_guidance else 1))]).to(device=device, dtype=dtype),
                        "aspect_ratio": torch.tensor([aspect_ratio for _ in range(batch_size * (2 if do_classifier_free_guidance else 1))]).to(device=device, dtype=dtype)
                    },
                    return_dict=False,
                )[0]
                
                # Classifier-free guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # Compute previous noisy sample
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                
                # Call callback, if provided
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)
                
                progress_bar.update()
                
                if XLA_AVAILABLE:
                    xm.mark_step()
        
        # 11. Final decoding
        if output_type == "latent":
            return latents
            
        images = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        images = self.image_processor.postprocess(images, output_type=output_type)
        
        if not return_dict:
            return (images,)
        
        return ImagePipelineOutput(images=images)
    
    def prepare_extra_step_kwargs(self, generator, eta):
        """
        Prepare extra kwargs for the scheduler step.
        """
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
            
        return extra_step_kwargs
import os
import warnings
from typing import List, Optional, Union, Tuple, Dict, Any, Callable

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
from localdit.pipeline_t2i import LocalDiTTxt2ImgPipeline

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)

class LocalDiTImg2ImgPipeline(DiffusionPipeline, TextualInversionLoaderMixin, IPAdapterMixin, StableDiffusionLoraLoaderMixin, FromSingleFileMixin):
    """
    Pipeline for image-to-image generation using LocalDiT.
    
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
        # We can leverage the same loader as Txt2Img
        return LocalDiTTxt2ImgPipeline.from_pretrained(pretrained_model_path, **kwargs)
    
    # We share many methods with the Txt2Img pipeline
    encode_prompt = LocalDiTTxt2ImgPipeline.encode_prompt
    set_image_sizes = LocalDiTTxt2ImgPipeline.set_image_sizes
    prepare_extra_step_kwargs = LocalDiTTxt2ImgPipeline.prepare_extra_step_kwargs
    
    def prepare_latents_from_image(
        self, 
        image, 
        timestep, 
        batch_size, 
        dtype, 
        device, 
        generator=None
    ):
        """
        Prepare latents from input image.
        """
        if isinstance(image, Image.Image):
            image = [image]
        
        # PIL image to tensor conversion
        image_tensors = []
        for img in image:
            img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
            img_tensor = 2.0 * img_tensor - 1.0  # [-1, 1] normalization
            image_tensors.append(img_tensor)
        
        image = torch.cat(image_tensors).to(device=device, dtype=dtype)
        
        # Encode image to latent space
        init_latents = self.vae.encode(image).latent_dist.sample(generator=generator)
        init_latents = self.vae.config.scaling_factor * init_latents
        
        # Add noise based on strength
        noise = torch.randn(init_latents.shape, device=device, dtype=dtype, generator=generator)
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
        
        return init_latents
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.Tensor, Image.Image, List[torch.Tensor], List[Image.Image]] = None,
        target_height: Optional[int] = 1152,
        target_width: Optional[int] = 2048,
        strength: float = 0.8,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        max_sequence_length: int = 120,
    ):
        """
        Generate images from a reference image and a text prompt using LocalDiT.
        
        Args:
            prompt (`str` or `List[str]`): Text prompt for image generation.
            image (`torch.Tensor` or `Image.Image` or `List[torch.Tensor]` or `List[Image.Image]`):
                Image or batch of images to use as reference for generation.
            target_height (`int`, *optional*, defaults to 1152): Target height for output image.
            target_width (`int`, *optional*, defaults to 2048): Target width for output image.
            strength (`float`, *optional*, defaults to 0.8): Strength of the refinement process.
            num_inference_steps (`int`, *optional*, defaults to 20): Number of refinement steps.
            guidance_scale (`float`, *optional*, defaults to 7.0): Guidance scale for classifier-free guidance.
            negative_prompt (`str` or `List[str]`, *optional*): Negative text prompt for classifier-free guidance.
            num_images_per_prompt (`int`, *optional*, defaults to 1): Number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0): Corresponds to parameter eta (η) from DDIM paper.
            generator (`torch.Generator`, *optional*): A torch generator for reproducibility.
            prompt_embeds (`torch.Tensor`, *optional*): Pre-computed text embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*): Pre-computed negative text embeddings.
            output_type (`str`, *optional*, defaults to "pil"): Output format ("pil", "latent" or "np").
            return_dict (`bool`, *optional*, defaults to True): Whether to return a dictionary or tuple.
            callback (`Callable`, *optional*): Callback function on each step.
            callback_steps (`int`, *optional*, defaults to 1): Frequency of callback function.
            max_sequence_length (`int`, *optional*, defaults to 120): Maximum text length for tokenization.
        
        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: An object containing the generated images.
        """
        # 0. Default values and basic checks
        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should be in [0.0, 1.0] but is {strength}")
        
        if image is None:
            raise ValueError("The `image` input cannot be None.")
            
        if target_height % self.vae_scale_factor != 0 or target_width % self.vae_scale_factor != 0:
            raise ValueError(f"target_height and target_width must be divisible by {self.vae_scale_factor}")
            
        # 1. Process inputs
        device = self._execution_device
        dtype = next(self.transformer.parameters()).dtype
        
        # 2. Preprocess image
        if not isinstance(image, torch.Tensor):
            if isinstance(image, Image.Image):
                image = [image]
                
            if isinstance(image[0], Image.Image):
                # Resize images to target dimensions
                resized_images = []
                for img in image:
                    resized_img = img.resize((target_width, target_height), Image.LANCZOS)
                    resized_images.append(resized_img)
                image = resized_images
            
        # 3. Encode prompt
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
        
        # 4. Set image sizes for attention processors
        self.set_image_sizes(target_height, target_width)
        
        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        # 6. Determine initial timestep based on strength
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = timesteps[t_start:]
        
        # 7. First timestep
        batch_size = 1 if isinstance(prompt, str) or prompt is None else len(prompt)
        batch_size = batch_size * num_images_per_prompt
        latent_timestep = timesteps[:1].repeat(batch_size)
        
        # 8. Prepare latents
        latents = self.prepare_latents_from_image(
            image=image,
            timestep=latent_timestep,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        
        # 9. Target aspect ratio
        aspect_ratio = target_width / target_height
        
        # 10. Prepare extra kwargs for the scheduler step
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        # 11. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        
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
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
                
                if XLA_AVAILABLE:
                    xm.mark_step()
        
        # 12. Final decoding
        if output_type == "latent":
            return latents
            
        images = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        images = self.image_processor.postprocess(images, output_type=output_type)
        
        if not return_dict:
            return (images,)
        
        return ImagePipelineOutput(images=images)
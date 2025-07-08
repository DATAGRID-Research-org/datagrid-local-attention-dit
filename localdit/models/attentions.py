from typing import Optional
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from diffusers.models.attention import Attention
from diffusers.utils.deprecation_utils import deprecate

class SE(nn.Module):
    """
    Squeeze and Excitation block
    """
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class FeatExtract(nn.Module):
    """
    Feature extraction block
    """
    def __init__(self, dim, keep_dim=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.GELU(),
            SE(dim, dim),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
        )
        if not keep_dim:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.keep_dim = keep_dim

    def forward(self, x):
        # Move conv to same device as input
        if next(self.conv.parameters()).device != x.device:
            self.conv = self.conv.to(x.device)
            
        x = x.contiguous()
        x = x + self.conv(x)
        if not self.keep_dim:
            if not hasattr(self, 'pool_device') or self.pool.device != x.device:
                self.pool = self.pool.to(x.device)
            x = self.pool(x)
        return x

class GCAttnProcessor2_0:
    """
    Global Context Attention Processor for DiT that's device-aware with window shift
    """
    def __init__(
            self,
            window_size=8,
            num_heads=8,
            use_global=True,
            shift_size=0,
            dim=1152,
            do_rope:bool=True,
            device=None,
            dtype=None):
        self.window_size = window_size
        self.num_heads = num_heads
        self.use_global = use_global
        self.shift_size = shift_size
        self.do_rope = do_rope
        # Lazy init for device compatibility
        self.initialized = False
        self.device = device
        self.dtype = dtype

        self.sin = torch.zeros((1, 1, 1, 1, 1))
        self.cos = torch.zeros((1, 1, 1, 1, 1))

        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError("GCAttnProcessor2_0 requires PyTorch 2.0+")
        
        # Global query generator with feature extraction
        if self.use_global:
            # Simple 2-layer feature extractor for global query generation
            self.query_feat_extract = nn.Sequential(
                FeatExtract(dim, keep_dim=True),
                FeatExtract(dim, keep_dim=True)
            ).to(self.device, dtype=self.dtype)
            
            # Initialize relative position bias for global attention
            self.global_rel_pos_bias = nn.Parameter(
                torch.zeros(
                    (2 * self.window_size - 1) * (2 * self.window_size - 1), self.num_heads,
                    device=self.device, dtype=self.dtype),
            )
            
            # Prepare relative position indices
            coords_h = torch.arange(self.window_size, device=self.device, dtype=self.dtype)
            coords_w = torch.arange(self.window_size, device=self.device, dtype=self.dtype)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
            coords_flatten = torch.flatten(coords, 1)
            
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size - 1
            relative_coords[:, :, 1] += self.window_size - 1
            relative_coords[:, :, 0] *= 2 * self.window_size - 1
            
            global_rel_pos_index = relative_coords.sum(-1).long()
            self.register_buffer("global_rel_pos_index", global_rel_pos_index, persistent=False)
            
            # Local projections for KV for global attention
            self.to_global_kv = nn.Linear(dim, dim * 2, bias=True).to(self.device, dtype=self.dtype)
            
        # Initialize regular window attention components
        self.local_rel_pos_bias = nn.Parameter(
            torch.zeros(
                (2 * self.window_size - 1) * (2 * self.window_size - 1), self.num_heads,
                device=self.device, dtype=self.dtype),   
        )
        
        coords_h = torch.arange(self.window_size, device=self.device, dtype=self.dtype)
        coords_w = torch.arange(self.window_size, device=self.device, dtype=self.dtype)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        
        local_rel_pos_index = relative_coords.sum(-1).long()
        self.register_buffer("local_rel_pos_index", local_rel_pos_index, persistent=False)
        
        # Initialize with Trunc Normal like in Swin
        # nn.init.trunc_normal_(self.local_rel_pos_bias, std=0.02)
        # if self.use_global:
        #     nn.init.trunc_normal_(self.global_rel_pos_bias, std=0.02)

    def create_position_encoding_2d(self, dim: int, device, dtype):
        # assert self.cos.shape == (1,1,1,1,1), "invalid"

        # Use self.height and self.width instead of seq_len
        height = self.height
        width = self.width

        theta = 1 / 10000 ** (4 * torch.arange(dim//4) / dim)

        pos_x = torch.arange(width)
        pos_y = torch.arange(height)

        theta = theta.reshape(1, dim//4)  # [1, dim//4]
        pos_x = pos_x.reshape(1, width, 1).repeat(height, 1, 1).reshape(height * width, 1)  # [height*width, 1]
        pos_y = pos_y.reshape(height, 1, 1).repeat(1, width, 1).reshape(height * width, 1)  # [height*width, 1]

        pos_theta = torch.cat([pos_x * theta, pos_y * theta], 1)  # [height*width, dim//2]
        cos = pos_theta.cos().reshape(height * width, dim//2, 1).repeat(1, 1, 2)  # [height*width, dim//2, 2]
        sin = pos_theta.sin().reshape(height * width, dim//2, 1).repeat(1, 1, 2)  # [height*width, dim//2, 2]

        self.cos = cos.reshape(1, 1, height * width, dim//2, 2).to(device, dtype=dtype)
        self.sin = sin.reshape(1, 1, height * width, dim//2, 2).to(device, dtype=dtype)

    def register_buffer(self, name, tensor, persistent=True):
        """Helper to register buffers across module calls"""
        if not hasattr(self, name):
            setattr(self, name, tensor)

    def _to_channel_first(self, x):
        """Tensor from (B, H, W, C) to (B, C, H, W)"""
        return x.permute(0, 3, 1, 2)
        
    def _to_channel_last(self, x):
        """Tensor from (B, C, H, W) to (B, H, W, C)"""
        return x.permute(0, 2, 3, 1)
        
    def _window_partition(self, x, window_size):
        """
        Args:
            x: (B, H, W, C)
            window_size: window size
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        h_windows = H // window_size
        w_windows = W // window_size
        
        # Reshape to group by windows
        x = x.view(B, h_windows, window_size, w_windows, window_size, C)
        # Permute and reshape to get windows as batch dimension
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows, h_windows, w_windows
        
    def _window_reverse(self, windows, window_size, H, W, B):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            window_size: Window size
            H, W: Original height and width
            B: Batch size
        Returns:
            x: (B, H, W, C)
        """
        h_windows = H // window_size
        w_windows = W // window_size
        C = windows.shape[-1]
        
        # Reshape to separate window dimensions
        x = windows.view(B, h_windows, w_windows, window_size, window_size, C)
        # Permute and reshape to original image dimensions
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
        return x

    def _get_attention_mask(self, h_windows, w_windows, padded_height, padded_width, device):
        """
        Create attention mask for SW-MSA if shift_size > 0
        Args:
            h_windows, w_windows: Number of windows in height/width
            padded_height, padded_width: Padded dimensions
            device: Computation device
        Returns:
            attn_mask: (num_windows, window_size*window_size, window_size*window_size)
        """
        if self.shift_size == 0:
            return None
            
        # Initialize attention mask
        img_mask = torch.zeros((1, padded_height, padded_width, 1), device=device)
        
        # Calculate slices for different regions
        h_slices = (slice(0, -self.window_size),
                   slice(-self.window_size, -self.shift_size),
                   slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                   slice(-self.window_size, -self.shift_size),
                   slice(-self.shift_size, None))
        
        # Mark different regions with different values
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        
        # Partition windows
        mask_windows, _, _ = self._window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        
        # Create attention mask based on region differences
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        # Convert to binary mask where different regions will have True (masked)
        attn_mask = attn_mask != 0
        
        return attn_mask

    def set_image_size(self, height, width, verbose=False):
        if verbose:
            print("Setting image size", height, width)
        self.height = int(height)
        self.width = int(width)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        mask_val = -10000.0  # Large negative value for masking attention
        residual = hidden_states
        
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            # If input is in image format [B, C, H, W], reshape to sequence format
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            # If already in sequence format [B, N, C]
            batch_size, sequence_length, _ = hidden_states.shape
            if self.height is None or self.width is None:
                height = width = int(math.sqrt(sequence_length))
            else:
                height, width = self.height, self.width
        
        # Apply group norm if provided
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        
        # Self-attention path
        if encoder_hidden_states is None:
            # Project to query, key, value
            qkv = attn.to_qkv(hidden_states)
            dim_per_head = qkv.shape[-1] // (3 * attn.heads)
            
            # Split QKV
            q, k, v = torch.split(qkv, qkv.shape[-1] // 3, dim=-1)
            
            # Reshape to [B, heads, N, head_dim]
            q = q.reshape(batch_size, -1, attn.heads, dim_per_head).permute(0, 2, 1, 3)
            k = k.reshape(batch_size, -1, attn.heads, dim_per_head).permute(0, 2, 1, 3)
            v = v.reshape(batch_size, -1, attn.heads, dim_per_head).permute(0, 2, 1, 3)
            
            # Apply norm to query and key if provided
            if attn.norm_q is not None:
                q = attn.norm_q(q)
            if attn.norm_k is not None:
                k = attn.norm_k(k)

            if self.do_rope:
                seq_len = q.shape[2]
                assert seq_len == self.width * self.height, f"invalid seq_len: {seq_len} != {self.width * self.height}"
                if self.cos.shape[2] != seq_len or self.cos.shape[3] != dim_per_head//2:
                    self.create_position_encoding_2d(dim_per_head, q.device, q.dtype)
                def apply_rotate(x):
                    x_even, x_odd = x.reshape(batch_size, attn.heads, seq_len, dim_per_head//2, 2).chunk(2, dim=4)
                    Rx = torch.cat([x_even, x_odd], 4) * self.cos + torch.cat([-x_odd, x_even], 4) * self.sin
                    Rx = Rx.reshape(batch_size, attn.heads, seq_len, dim_per_head)
                    return Rx
                q = apply_rotate(q)
                k = apply_rotate(k)
                
            # Reshape for window partitioning - to spatial layout [B, H, W, C]
            q_spatial = q.permute(0, 2, 1, 3).reshape(batch_size, height, width, attn.heads * dim_per_head)
            k_spatial = k.permute(0, 2, 1, 3).reshape(batch_size, height, width, attn.heads * dim_per_head)
            v_spatial = v.permute(0, 2, 1, 3).reshape(batch_size, height, width, attn.heads * dim_per_head)
            
            # Pad if needed to be divisible by window_size
            pad_h = (self.window_size - height % self.window_size) % self.window_size
            pad_w = (self.window_size - width % self.window_size) % self.window_size
            
            if pad_h > 0 or pad_w > 0:
                q_spatial = F.pad(q_spatial, (0, 0, 0, pad_w, 0, pad_h))
                k_spatial = F.pad(k_spatial, (0, 0, 0, pad_w, 0, pad_h))
                v_spatial = F.pad(v_spatial, (0, 0, 0, pad_w, 0, pad_h))
                
            padded_height = height + pad_h
            padded_width = width + pad_w
            
            # Apply window shift if shift_size > 0
            if self.shift_size > 0:
                # Shift the feature maps
                q_spatial = torch.roll(q_spatial, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                k_spatial = torch.roll(k_spatial, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                v_spatial = torch.roll(v_spatial, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            
            # Calculate attention mask for shifted windows
            h_windows = padded_height // self.window_size
            w_windows = padded_width // self.window_size
            attn_mask = self._get_attention_mask(h_windows, w_windows, padded_height, padded_width, q_spatial.device)
            # The attention mask has shape [num_windows, window_size*window_size, window_size*window_size]
            
            # Partition windows
            q_windows, h_windows, w_windows = self._window_partition(q_spatial, self.window_size)
            k_windows, _, _ = self._window_partition(k_spatial, self.window_size)
            v_windows, _, _ = self._window_partition(v_spatial, self.window_size)
            
            # Process with window attention
            window_size_sq = self.window_size * self.window_size
            attn_dim = q_windows.shape[-1]
            
            # Reshape for attention computation
            q_windows = q_windows.reshape(-1, window_size_sq, attn_dim)
            k_windows = k_windows.reshape(-1, window_size_sq, attn_dim)
            v_windows = v_windows.reshape(-1, window_size_sq, attn_dim)
            
            # Number of window groups (B * num_windows)
            num_groups = q_windows.shape[0]
            
            # Local window attention
            q_local = q_windows.reshape(num_groups, window_size_sq, attn.heads, dim_per_head)
            k_local = k_windows.reshape(num_groups, window_size_sq, attn.heads, dim_per_head)
            v_local = v_windows.reshape(num_groups, window_size_sq, attn.heads, dim_per_head)
            
            # Transpose for attention calculation
            q_local = q_local.permute(0, 2, 1, 3)  # [B*windows, heads, window_size_sq, head_dim]
            k_local = k_local.permute(0, 2, 1, 3)
            v_local = v_local.permute(0, 2, 1, 3)
            
            # Compute attention
            attn_scale = dim_per_head ** -0.5
            attn_local = (q_local @ k_local.transpose(-2, -1)) * attn_scale
            
            # Add relative position bias
            relative_position_bias = self.local_rel_pos_bias[self.local_rel_pos_index.view(-1)].view(
                window_size_sq, window_size_sq, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
            attn_local = attn_local + relative_position_bias
            
            # Apply attention mask for shifted windows
            if attn_mask is not None:
                # Reshape attention mask to match attention dimensions
                # attn_local is [num_groups, num_heads, window_size*window_size, window_size*window_size]
                # attn_mask is [num_windows, window_size*window_size, window_size*window_size]
                # Need to expand to match dimensions
                attn_mask = attn_mask.unsqueeze(1)  # [num_windows, 1, window_size*window_size, window_size*window_size]
                # Make sure num_windows matches num_groups
                if attn_mask.size(0) != num_groups:
                    attn_mask = attn_mask.repeat(batch_size, 1, 1, 1)  # Repeat for each batch item
                attn_local = attn_local.masked_fill(attn_mask, mask_val)
            
            # Apply softmax
            attn_local = F.softmax(attn_local, dim=-1)
            
            # Compute output
            output_local = attn_local @ v_local  # [B*windows, heads, window_size_sq, head_dim]
            
            # Global attention if enabled
            if self.use_global:
                # Generate global query
                x_global = self._to_channel_first(q_spatial)
                x_global = self.query_feat_extract(x_global)
                x_global = self._to_channel_last(x_global)
                
                # Global KV
                kv = self.to_global_kv(k_windows.reshape(num_groups, window_size_sq, attn_dim))
                k_global, v_global = torch.split(kv, kv.shape[-1] // 2, dim=-1)
                
                # Reshape for attention
                k_global = k_global.reshape(num_groups, window_size_sq, attn.heads, dim_per_head).permute(0, 2, 1, 3)
                v_global = v_global.reshape(num_groups, window_size_sq, attn.heads, dim_per_head).permute(0, 2, 1, 3)
                
                # Extract global query from the first window of each batch
                q_global = x_global.reshape(batch_size, padded_height, padded_width, attn_dim)
                q_global_windows, _, _ = self._window_partition(q_global, self.window_size)
                q_global = q_global_windows.reshape(batch_size * h_windows * w_windows, window_size_sq, attn_dim)
                q_global = q_global.reshape(num_groups, window_size_sq, attn.heads, dim_per_head).permute(0, 2, 1, 3)
                
                # Global attention
                attn_global = (q_global @ k_global.transpose(-2, -1)) * attn_scale
                
                # Add relative position bias
                relative_position_bias = self.global_rel_pos_bias[self.global_rel_pos_index.view(-1)].view(
                    window_size_sq, window_size_sq, -1)
                relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
                attn_global = attn_global + relative_position_bias
                
                # Apply attention mask for shifted windows
                if attn_mask is not None:
                    # Make sure dimensions match
                    if attn_mask.size(0) != num_groups:
                        attn_mask = attn_mask.repeat(batch_size, 1, 1, 1)
                    attn_global = attn_global.masked_fill(attn_mask, mask_val)
                
                # Apply softmax
                attn_global = F.softmax(attn_global, dim=-1)
                
                # Compute global output
                output_global = attn_global @ v_global
                
                # Combine local and global
                output = (output_local + output_global) / 2
            else:
                output = output_local
            
            # Reshape back
            output = output.permute(0, 2, 1, 3).reshape(num_groups, window_size_sq, attn_dim)
            
            # Window reverse
            output = output.reshape(num_groups, self.window_size, self.window_size, attn_dim)
            output_spatial = self._window_reverse(output, self.window_size, padded_height, padded_width, batch_size)
            
            # Reverse the shift if shift_size > 0
            if self.shift_size > 0:
                output_spatial = torch.roll(output_spatial, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            
            # Remove padding
            if pad_h > 0 or pad_w > 0:
                output_spatial = output_spatial[:, :height, :width, :]
            
            # Reshape back to sequence format
            hidden_states = output_spatial.reshape(batch_size, height * width, attn_dim)
            
        else:
            # Cross-attention path
            q = attn.to_q(hidden_states)
            
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
                
            kv = attn.to_kv(encoder_hidden_states)
            k, v = torch.split(kv, kv.shape[-1] // 2, dim=-1)
            
            inner_dim = k.shape[-1]
            head_dim = inner_dim // attn.heads
            
            q = q.reshape(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            k = k.reshape(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            v = v.reshape(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            
            # Apply norm if provided
            if attn.norm_q is not None:
                q = attn.norm_q(q)
            if attn.norm_k is not None:
                k = attn.norm_k(k)
            
            # Compute attention with scaled_dot_product_attention
            hidden_states = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=0.0,
            )
            
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        
        # Linear projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)  # dropout
        
        # Reshape back if input was 4D
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        
        # Residual connection
        hidden_states = hidden_states + residual
        
        # Apply rescale factor
        hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states
    
# Helper function to apply GC attention to a model
def apply_gc_attn_to_dit(model, window_size=8, use_global=True):
    """
    Apply Global Context Attention to all self-attention blocks in a DiT model
    
    Args:
        model: A DiT model
        window_size: Size of attention windows
        use_global: Whether to use global context
    
    Returns:
        model: The model with GC attention processors applied
    """
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            # Get number of heads from the module
            num_heads = module.heads
            
            # Only apply to self-attention blocks
            if "attn1" in name:
                module.set_processor(GCAttnProcessor2_0(
                    window_size=window_size,
                    num_heads=num_heads,
                    use_global=use_global
                ))
    
    return model

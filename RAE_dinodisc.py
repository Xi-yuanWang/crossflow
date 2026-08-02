'''
Modified from https://github.com/bytetriper/RAE/src/disc/dinodisc.py
Class-conditional variant inspired by StyleGAN-XL's projection discriminator.
'''
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import SpectralNorm
from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTModel
from torch.autograd import Function

def spectral_linear(in_features: int, out_features: int) -> nn.Linear:
    """Create a Linear layer with spectral normalization."""
    layer = nn.utils.spectral_norm(nn.Linear(in_features, out_features))
    # SpectralNorm.apply(layer)
    return layer

def spectral_conv(in_features: int, out_features: int, kernel_size: int) -> nn.Conv2d:
    """Create a conv layer with spectral normalization."""
    layer = nn.utils.spectral_norm(nn.Conv2d(in_features, out_features, kernel_size, padding=kernel_size//2))
    # SpectralNorm.apply(layer)
    return layer

class ResBlock(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x):
        return (x + self.block(x)) * (2**-0.5)

def make_convhead(channels: int, outdim: int) -> nn.Module:
    """Per-token MLP head: Linear → LeakyReLU → Linear → 1, all spectral-normed.

    Operates independently on each token, then the caller mean-pools over tokens.
    """
    return nn.Sequential(
        spectral_conv(channels, channels, 1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        ResBlock(nn.Sequential(
            spectral_conv(channels, channels, 3), nn.LeakyReLU(negative_slope=0.2, inplace=True))),
        spectral_conv(channels, outdim, 1),
    )


def make_head(channels: int, outdim: int) -> nn.Module:
    """Per-token MLP head: Linear → LeakyReLU → Linear → 1, all spectral-normed.

    Operates independently on each token, then the caller mean-pools over tokens.
    """
    return nn.Sequential(
        spectral_linear(channels, channels),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        ResBlock(nn.Sequential(
            spectral_linear(channels, channels), nn.LeakyReLU(negative_slope=0.2, inplace=True))),
        spectral_linear(channels, outdim),
    )


class DinoDisc(nn.Module):
    """Class-conditional DINO-based discriminator (StyleGAN-XL style).

    Extracts intermediate hidden states from a frozen DINOv3 ViT backbone
    and feeds them through learnable per-token MLP heads with spectral norm.
    Each head produces a per-token scalar, which is mean-pooled to get a
    single logit per layer.

    **Class conditioning** follows the projection discriminator design
    (Miyato & Koyama, 2018) used in StyleGAN-XL:
      - A shared, spectrally-normalised class embedding table maps each
        class index to a vector of size ``embed_dim``.
      - For each DINO layer head, a spectral-normed linear projection maps
        the class embedding to a scalar bias that is added to the
        unconditional logit.  This lets the discriminator learn
        class-specific decision boundaries without inflating the head
        capacity.
    """

    def __init__(
        self,
        device: torch.device,
        dino_model_path: str = "./dinov3",
        key_depths: Tuple[int, ...] = (2, 5, 8, 11),
        num_classes: int = 1000,
        class_embed_dim: int = 512,
    ):
        super().__init__()

        # --- Frozen DINO backbone (not registered as submodule) ---
        dino = DINOv3ViTModel.from_pretrained(dino_model_path)
        dino.eval()
        for param in dino.parameters():
            param.requires_grad_(False)
        self.dino = dino

        num_layers = dino.config.num_hidden_layers  # 12
        self.key_depths = tuple(d for d in key_depths if d < num_layers)
        # 1 CLS + num_register_tokens (4) = 5 prefix tokens to skip
        self.num_prefix_tokens = 1 + dino.config.num_register_tokens

        # ImageNet normalization for input in [-1, 1]:
        #   x_norm = ((x+1)/2 - mean) / std = (0.5/std)*x + (0.5-mean)/std
        mean = torch.tensor((0.485, 0.456, 0.406))
        std = torch.tensor((0.229, 0.224, 0.225))
        self.register_buffer("x_scale", (0.5 / std).reshape(1, 3, 1, 1))
        self.register_buffer("x_shift", ((0.5 - mean) / std).reshape(1, 3, 1, 1))

        dino_C = dino.config.hidden_size
        num_heads = len(self.key_depths) + 1  # +1 for the final layer

        # --- Unconditional per-token MLP heads (same as before) ---
        self.heads = nn.ModuleList(
            [make_convhead(dino_C, 1) for _ in range(num_heads)]
        )

        # --- Class-conditional projection (StyleGAN-XL style) ---

        # Per-head projection: maps the CLS (global) token to the same
        # space as the class embedding, then inner-product gives the
        # conditional logit contribution.
        self.cls_proj = nn.ModuleList(
            [make_head(dino_C, num_classes) for _ in range(num_heads)]
        )

        from diffaug import DiffAug
        self.diffaug = DiffAug()

    def _get_acts(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Extract DINO features and return intermediate activations.

        Args:
            x: Images in [-1, 1], shape (B, 3, H, W).
        Returns:
            Tuple of (patch_activations, cls_activations):
              - patch_activations: list of (B, N, C) patch token features.
              - cls_activations:   list of (B, C) CLS (global) token features.
        """
        pixel_values = x * self.x_scale + self.x_shift

        outputs = self.dino(pixel_values=pixel_values, output_hidden_states=True)

        hidden_states = outputs.hidden_states

        patch_activations: List[torch.Tensor] = []
        cls_activations: List[torch.Tensor] = []
        for d in self.key_depths:
            h = hidden_states[d + 1]
            cls_activations.append(h[:, 0, :])                       # (B, C)
            patch_activations.append(h[:, self.num_prefix_tokens:, :])  # (B, N, C)

        final = hidden_states[-1]
        cls_activations.insert(0, final[:, 0, :])                       # (B, C)
        patch_activations.insert(0, final[:, self.num_prefix_tokens:, :])  # (B, N, C)
        return patch_activations, cls_activations

    def _get_logits(
        self,
        patch_acts: List[torch.Tensor],
        cls_acts: List[torch.Tensor],
        class_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply per-token MLP heads, mean-pool, and add class projection.

        The unconditional logit is computed from **patch tokens** (per-token
        MLP → mean-pool).  The conditional logit uses the **CLS (global)
        token** — projected and inner-producted with the class embedding —
        following the StyleGAN-XL projection discriminator design.

        Args:
            patch_acts: List of tensors, each (B, N, C) — patch tokens.
            cls_acts:   List of tensors, each (B, C) — CLS global tokens.
            class_labels: Integer class indices of shape (B,).
                          If *None*, only unconditional logits are returned.
        Returns:
            Concatenated logits of shape (B, num_heads).
        """
        # Pre-compute class embedding once (shared across heads)
        logits = []
        for idx, (head, p_act, c_act) in enumerate(zip(self.heads, patch_acts, cls_acts)):
            # Unconditional branch: per-token MLP on patch tokens → mean-pool
            uncond = head(p_act.transpose(1, 2).unflatten(2, (16, 16))).squeeze(-1)           # (B, N)
            uncond = uncond.mean(dim=(2, 3))  # (B, 1)
            proj = self.cls_proj[idx](c_act)       # (B, class_embed_dim)
            cond = torch.gather(proj, dim=1, index=class_labels.reshape(-1, 1)) # (B, 1)
            logits.append(uncond)
            logits.append(cond)
        return torch.cat(logits, dim=1)  # (B, 2*num_heads)

    def forward(
        self,
        real: torch.Tensor = None,
        fake: torch.Tensor = None,
        class_labels: torch.Tensor = None,
        act_cache: tuple = None,
        mode: str = "g_loss",
    ):
        """Two-mode forward for DDP compatibility.

        Mode ``"g_loss"`` (generator step):
            Extracts DINO activations for *real* and *fake*, computes
            ``g_loss = -mean(logits_fake)``, and returns
            ``(g_loss, act_cache)`` where *act_cache* stores detached
            activations **and class labels** for the subsequent
            discriminator step.

        Mode ``"d_loss"`` (discriminator step):
            Takes *act_cache* produced by the generator step and computes
            the hinge discriminator loss.  Returns a scalar ``d_loss``.

        Args:
            real: Real images in [-1, 1], shape (B, 3, H, W).
            fake: Fake images in [-1, 1], shape (B, 3, H, W).
            class_labels: Integer class indices, shape (B,).
            act_cache: Cached (act_fake, act_real, class_labels) from g_loss step.
            mode: ``"g_loss"`` or ``"d_loss"``.
        """
        if mode == "g_loss":
            patch_fake, cls_fake = self._get_acts(self.diffaug.aug(fake))   # grad graph back to generator
            with torch.no_grad():
                patch_real, cls_real = self._get_acts(self.diffaug.aug(real))  # fully detached

            logits_fake = self._get_logits(patch_fake, cls_fake, class_labels)
            g_loss = -logits_fake.mean()

            # Cache detached activations + labels for the d_loss step
            act_cache = (
                [a.detach() for a in patch_fake],
                [a.detach() for a in cls_fake],
                patch_real,
                cls_real,
                class_labels.detach() if class_labels is not None else None,
            )
            return g_loss, act_cache

        elif mode == "d_loss":
            patch_fake, cls_fake, patch_real, cls_real, cached_labels = act_cache
            logits_real = self._get_logits(patch_real, cls_real, cached_labels)
            logits_fake = self._get_logits(patch_fake, cls_fake, cached_labels)
            d_loss = 0.5 * (
                F.relu(1.0 - logits_real).mean()
                + F.relu(1.0 + logits_fake).mean()
            )
            return d_loss

        else:
            raise ValueError(f"Unknown mode: {mode!r}. Use 'g_loss' or 'd_loss'.")
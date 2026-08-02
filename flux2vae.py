# export BNB_CUDA_VERSION=130
from PIL import Image
import torch
from diffusers import AutoencoderKLFlux2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
class Fluxv2vae_wrapper(nn.Module):
    def __init__(self, embed_dim, ch_mult, use_variational=True, ckpt_path=None, model_type='vavae'):
        super().__init__()
        self.vae = AutoencoderKLFlux2.from_pretrained("./fluxv2vae/", subfolder="vae")
        self.out_channels = 128
        self.in_channels = 3

    def encode(self, x):
        return self.vae.encode(x).latent_dist.sample()

    def decode(self, z):
        x = self.vae.decode(z).sample()
        return x

    def forward(self, inputs, return_recon=False):
        z = self.encode(inputs)
        if return_recon:
            dec = self.decode(z)
            return F.pixel_unshuffle(z, 2).flatten(-2, -1).transpose(-2, -1), dec
        return F.pixel_unshuffle(z, 2).flatten(-2, -1).transpose(-2, -1)
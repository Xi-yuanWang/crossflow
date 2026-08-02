from diffusers.models import AutoencoderKL
import torch.nn as nn
import torch.nn.functional as F


class SDvae_wrapper(nn.Module):
    def __init__(self, embed_dim=None, ch_mult=None, use_variational=True, ckpt_path=None, model_type='sdvae'):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained("./sd-vae-ft-mse")
        # SD VAE: 4 latent channels, 8x spatial downsampling.
        # After pixel_unshuffle(2): 4 * 2^2 = 16 channels, 16x effective downsampling.
        self.out_channels = 16
        self.in_channels = 3

    def encode(self, x):
        return self.vae.encode(x).latent_dist.sample()

    def decode(self, z):
        x = self.vae.decode(z).sample
        return x

    def forward(self, inputs, return_recon=False):
        z = self.encode(inputs)
        tokens = F.pixel_unshuffle(z, 2).flatten(-2, -1).transpose(-2, -1)
        if return_recon:
            dec = self.decode(z)
            return tokens, dec
        return tokens

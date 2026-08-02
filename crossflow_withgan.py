
from math import sqrt
import torch, pdb
import torch.nn as nn
import torch, math
from torch import nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Mlp
from transformer_utils import VisionRotaryEmbeddingFast, RMSNorm, GaussianFourierEmbedding,NormAttention, get_2d_sincos_pos_embed
import torch.nn.functional as F
from typing import *

def normalize(x, eps = 1.):
    return (1 + eps)**0.5 * x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function

class ClipGradFunction(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        norm = torch.norm(grad_output.flatten(1, -1), dim=1)
        clipednorm = torch.clamp_max(norm, 1./grad_output.shape[0])
        grad_output = grad_output * ((clipednorm / (norm + 1e-3)).reshape(-1, 1, 1, 1))
        return grad_output


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, token_num, frequency_embedding_size=256, twocond: bool=False):
        super().__init__()
        self.token_num = token_num
        self.hidden_size = hidden_size
        self.twocond = twocond
        self.mlp = nn.Sequential(
            nn.Linear(2*frequency_embedding_size if twocond else frequency_embedding_size, 2*frequency_embedding_size, bias=True),
            nn.SiLU(inplace=True),
            nn.Linear(2*frequency_embedding_size, self.token_num*hidden_size, bias=True),
        )

        self.frequency_embedding_size = frequency_embedding_size
        half = frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        )
        self.register_buffer("freqs", freqs)

    def timestep_embedding(self, t, dim,):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py        
        args = t[:, None].float() * self.freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t1, t2):
        t2emb = self.timestep_embedding(t2.reshape(-1), self.frequency_embedding_size).reshape(t2.shape[0], -1)
        if self.twocond:
            t1emb = self.timestep_embedding(t2.reshape(-1) - t1.reshape(-1), self.frequency_embedding_size).reshape(t1.shape[0], -1)
            return self.mlp(torch.concat((t1emb, t2emb), dim=-1)).reshape(t1.shape[0], self.token_num, self.hidden_size)
        else:
            return self.mlp(t2emb).reshape(t1.shape[0], self.token_num, self.hidden_size)


class LabelCfgEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, token_num):
        super().__init__()
        self.token_num = token_num
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(1001, hidden_size*token_num)

    def forward(self, label):
        labelemb = self.embedding(label.flatten()).unflatten(-1, (self.token_num, self.hidden_size))
        return labelemb


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class Block(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        scale_shift = False,
        conddim=None,
    ):
        super().__init__()
        if conddim is None:
            conddim = hidden_size

        # Initialize normalization layers
        self.norm1 = RMSNorm(hidden_size, learnable_scale = not scale_shift)
        self.norm2 = RMSNorm(hidden_size, learnable_scale = not scale_shift)

        # Initialize attention layer
        self.attn = NormAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=False,
            use_rmsnorm=False,
        )

        # Initialize MLP layer
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        def approx_silu(): return nn.SiLU(inplace=True)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_silu,
            drop=0
        )
        if scale_shift:
            self.scale_shift = nn.Linear(conddim, 6 * hidden_size, bias=True)

    def forward(self, x, feat_rope = None, c = None):
        if c is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.scale_shift(c).chunk(6, dim=1)
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        else:
            x = x + self.attn(self.norm1(x), rope=feat_rope) 
            x = x + self.mlp(self.norm2(x))
        return x

def pix2seq(x, patch_size):
    # pixel_unshuffle: (B, C, H, W) -> (B, C*p*p, H/p, W/p)
    x = F.pixel_unshuffle(x, patch_size)
    grid_h, grid_w = x.shape[2], x.shape[3]
    # Reshape to (B, N, C*p*p) and project to hidden_size
    x = x.flatten(2, -1).permute(0, 2, 1)
    return x, grid_h, grid_w

def seq2pix(x, patch_size, grid_h, grid_w):
    # Reshape to (B, C*p*p, H/p, W/p)
    x = x.permute(0, 2, 1).reshape(x.shape[0], -1, grid_h, grid_w)
    # pixel_shuffle: (B, C*p*p, H/p, W/p) -> (B, C, H, W)
    x = F.pixel_shuffle(x, patch_size)
    return x

class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_channels, hidden_size, bias):  
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.bias = bias
        self.rawsize = in_channels * patch_size * patch_size

    def forward(self, x):
        assert x.shape[2] % self.patch_size == 0
        assert x.shape[3] % self.patch_size == 0
        rawx, grid_h, grid_w = pix2seq(x, self.patch_size)
        x = rawx
        pos_embed = torch.zeros_like(x) #get_2d_sincos_pos_embed(self.hidden_size, grid_h, grid_w, x.device).to(x)
        # x_shape keeps (B, hidden_size, grid_h, grid_w) for downstream compatibility
        x_shape = (x.shape[0], self.hidden_size, grid_h, grid_w)
        x = x + pos_embed
        return x, rawx, x_shape, pos_embed    

class AdaLNOut(nn.Module):
    def __init__(
        self,
        hidden_size,
        out_size,
        scale_shift = False,
        conddim=None,
    ):
        super().__init__()
        if conddim is None:
            conddim = hidden_size

        # Initialize normalization layers
        self.norm1 = RMSNorm(hidden_size, learnable_scale = False)

        self.lin = nn.Linear(hidden_size, out_size)

        if scale_shift:
            self.scale_shift = nn.Linear(conddim, 2*hidden_size, bias=True)
            nn.init.zeros_(self.scale_shift.weight)
            nn.init.zeros_(self.scale_shift.bias)

    def forward(self, x, c = None):
        if c is not None:
            shift_msa, scale_msa = self.scale_shift(c).chunk(2, dim=1)
            x = self.lin(modulate(self.norm1(x), shift_msa, scale_msa))
        else:
            x = self.lin(self.norm1(x)) 
        return x

class RepProj(nn.Module):

    def __init__(self, hidden_size, code_dim, pca_path="vavae_pca.pth"):
        super().__init__()
        xmean, eigval, eigvec = torch.load(pca_path, map_location="cpu")
        self.weight = nn.Buffer(eigvec[:, -code_dim:]*torch.rsqrt(eigval[-code_dim:]))
        self.bias = nn.Buffer(- self.weight.T @ xmean)
    
    def forward(self, x):
        return x @ self.weight.to(x.dtype) + self.bias.to(x.dtype)

class ViT(nn.Module):
    def __init__(
            self,
            patch_size: int = 16,
            in_channels: int = 3,
            code_dim = 128,
            hidden_size: int =  1024 ,
            enc_depth: int = 12,
            dec_depth: int = 12,
            num_heads: int = 16,
            mlp_ratio: float=4.0,
            num_timetoken: int=8,
            scale_shift=False,
            use_label: bool = True,
            discriminator_depth: int = 3,
            twocond: bool = False,
            pca_path: str = "vavae_pca.pth"): 
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.scale_shift = scale_shift
        self.use_label = use_label

        self.indim = patch_size ** 2 * in_channels

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=hidden_size // num_heads // 2,
        )
        
        self.num_timetoken = num_timetoken

        self.dec_in = nn.Linear(code_dim, hidden_size)
        self.t_emb = TimestepEmbedder(self.hidden_size, self.num_timetoken + 1 if scale_shift else self.num_timetoken, twocond=twocond)
        if self.use_label:
            self.label_emb = LabelCfgEmbedder(self.hidden_size, self.num_timetoken + 1 if scale_shift else self.num_timetoken)
            self.num_timetoken = self.num_timetoken * 2
        self.dec_zout = AdaLNOut(hidden_size, code_dim, scale_shift=scale_shift, conddim=self.hidden_size)
        self.dec_out = AdaLNOut(hidden_size, patch_size ** 2 * in_channels, scale_shift=scale_shift, conddim=self.hidden_size)
        self.blocks1 = nn.ModuleList([Block(hidden_size, num_heads, mlp_ratio,  scale_shift=scale_shift, conddim=self.hidden_size) for _ in range(enc_depth)])
        self.blocks2 = nn.ModuleList([Block(hidden_size, num_heads, mlp_ratio,  scale_shift=scale_shift, conddim=self.hidden_size) for _ in range(dec_depth)])

        if discriminator_depth > 0:
            self.blocks_discriminator = nn.ModuleList([Block(hidden_size, num_heads, mlp_ratio,  scale_shift=False, conddim=self.hidden_size) for _ in range(discriminator_depth)])
            self.head_discriminator = AdaLNOut(hidden_size, 1, scale_shift=False)
            self.linin_discriminator = nn.Linear(patch_size ** 2 * in_channels, hidden_size)

        self.rep_reducer = RepProj(hidden_size, code_dim, pca_path=pca_path)

        self.initialize_weights()

    def initialize_weights(self, xavier_uniform_init: bool = False):
        # Zero-out adaLN modulation layers in LightningDiT blocks:
        for net in [self.blocks1, self.blocks2]:
            for block in net:
                nn.init.constant_(block.attn.proj.weight, 0)
                nn.init.constant_(block.mlp.fc2.weight, 0)
                if hasattr(block, 'scale_shift'):
                    nn.init.constant_(block.scale_shift.weight, 0)
        nn.init.constant_(self.dec_out.lin.weight, 0)
        nn.init.constant_(self.dec_zout.lin.weight, 0)
        nn.init.constant_(self.t_emb.mlp[-1].weight, 0)
        nn.init.normal_(self.t_emb.mlp[-1].bias, std=0.1)


    def forward(self, code, t1, t2, label, orix=None, ganleak: float=1.):
        c = self.t_emb(t1, t2)
        if self.scale_shift:
            adac, c = c[:, 0], c[:, 1:]
        else:
            adac = None
        if self.use_label:
            l = self.label_emb(label)
            if self.scale_shift:
                adal, l = l[:, 0], l[:, 1:]
                adac = adac + adal
            else:
                adal = None
            c = torch.concat((c, l), dim=1)
        feat_rope = self.feat_rope(16, 16, code.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            x = self.dec_in(code)
            x = torch.concat((c, x), dim=1)
            for blk in self.blocks1:
                x = blk(x, feat_rope, adac)
            z = self.dec_zout(x[:, self.num_timetoken:], adac)
            for blk in self.blocks2:
                x = blk(x, feat_rope, adac)
            x = self.dec_out(x[:, self.num_timetoken:], adac)
            x = seq2pix(x, 16, 16, 16)

        if orix is not None:
            B = orix.shape[0]
            allx = torch.concat((orix, ganleak*(x.detach() - x) + x.detach()), dim=0)
            allx = ClipGradFunction.apply(allx) 
            # (2*B, C, H, W), generator and discriminator's optimization direction is opposite
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                allx, _, _ = pix2seq(allx, 16)
                allx = torch.concat((torch.concat((c.detach(), c.detach()), dim=0), self.linin_discriminator(allx)), dim=1)
                for blk in self.blocks_discriminator:
                    allx = blk(allx, feat_rope)
                allx = self.head_discriminator(allx[:, self.num_timetoken:]) # (2*B, L, 1)
            posx, negx = allx[:B, :, 0], allx[B:, :, 0] # (B, L), (B, L)
            return - F.logsigmoid(posx).mean(), - F.logsigmoid(-negx).mean()
        else:
            return x, z
        
if __name__ == "__main__":
    vit = ViT()
    vit.cuda()
    with torch.no_grad():
        dec_x = vit(torch.randn(10, 128, 16, 16).float().cuda(), torch.randn(10).float().cuda(), torch.randn(10).float().cuda())
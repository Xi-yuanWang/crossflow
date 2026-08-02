
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from perceptual.lpips import LPIPS
from perceptual.dinoperc import DINO_Perceptual_Loss
import torchvision
from einops import rearrange


def create_custom_forward(module):
    def custom_forward(*inputs):
        return module(*inputs)
    return custom_forward

import random


def random_crop_and_resize(img, tar_img, scale=(0.6, 1.0), target_size = 224):
    C, H, W = img.shape[1:]
    area = H * W
    target_area = random.uniform(*scale) * area
    aspect_ratio = random.uniform(3/4, 4/3)
    h = int((target_area / aspect_ratio) ** 0.5)
    w = int((target_area * aspect_ratio) ** 0.5)
    h = min(h, H)
    w = min(w, W)
    top = random.randint(0, H - h)
    left = random.randint(0, W - w)
    crop = img[:, :, top:top+h, left:left+w]
    crop_tar = tar_img[:, :, top:top+h, left:left+w]

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=False):
        resized = F.interpolate(
            crop.float(),
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False
        )
        resized_tar = F.interpolate(
            crop_tar.float(),
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False
        )
    return resized, resized_tar

class PerceptualLoss(nn.Module):
    def __init__(self, use_l1=False, start_layer: int = 1):
        super().__init__()
        # self.perceptual_loss = LPIPS().eval()
        # for param in self.perceptual_loss.parameters():
        #     param.requires_grad = False
        self.perceptual_loss2 = DINO_Perceptual_Loss(use_l1=use_l1, start_layer=start_layer).eval()

    def forward(self, inputs: torch.Tensor, reconstructions: torch.Tensor, usecrop: bool=True):
        if usecrop:
            inputs, reconstructions = random_crop_and_resize(inputs, reconstructions)
        loss1, loss2, loss3 = self.perceptual_loss2(inputs, reconstructions)
        return loss1.mean() + loss2.mean() + loss3.mean()

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

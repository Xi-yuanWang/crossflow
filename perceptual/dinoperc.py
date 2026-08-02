import torch
import torch.nn as nn
import torchvision
from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTModel
F = torch.nn.functional
import random

class DINO_Perceptual_Loss(nn.Module):
    def __init__(self, use_l1=False, start_layer: int=1):
        super().__init__()
        self.dinov3 = DINOv3ViTModel.from_pretrained("./dinov3")
        for param in self.dinov3.parameters():
            param.requires_grad = False
        self.num_layers = self.dinov3.config.num_hidden_layers
        self.start_layer = start_layer
        self.img_imagenet_normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=(0.229, 0.224, 0.225), inplace=True)
        self.use_l1 = use_l1

    def _get_intermediate_layers(self, x):
        """Extract intermediate layer features similar to the original get_intermediate_layers."""
        outputs = self.dinov3(pixel_values=x, output_hidden_states=True)
        # hidden_states includes embedding output + all layer outputs, so [1:] gives layer outputs
        hidden_states = outputs.hidden_states[self.start_layer:]  # tuple num_layers x (B, seq_len, dim)
        hidden_states = torch.stack(hidden_states, dim=1)  # (B, numlayer, seq_len, dim)
        # Return the first n layers, excluding the CLS token (index 0) and 4 register tokens.
        return hidden_states[:, :, 0, :], hidden_states[:, :, 5:, :]

    def forward(self, tar, pred):
        with torch.no_grad():
            tar_input = self.img_imagenet_normalize(tar.contiguous().float().cuda() * 0.5 + 0.5)
            tarcls, tar_list = self._get_intermediate_layers(tar_input) # (num_layers, B, seq_len, dim)
        pred_input = self.img_imagenet_normalize(pred.contiguous().float().cuda() * 0.5 + 0.5)
        predcls, pred_list = self._get_intermediate_layers(pred_input) # (num_layers, B, seq_len, dim)
        
        if self.use_l1:
            clsloss = torch.sqrt((predcls - tarcls).square().flatten(1, -1).mean(dim=1) + 1).mean(dim=0)
            losses = torch.sqrt((pred_input - tar_input).square().flatten(1, -1).mean(dim=1) + 1).mean(dim=0)
        else:
            clsloss = (predcls - tarcls).square().flatten(1, -1).mean(dim=1).mean(dim=0)
            losses = (pred_input - tar_input).square().flatten(1, -1).mean(dim=1).mean(dim=0)
        
        fp = F.normalize(pred_input, dim=-1)
        ft = F.normalize(tar_input, dim=-1)
        fp_gram = fp @ fp.permute(0, 1, 3, 2)
        tp_gram = ft @ ft.permute(0, 1, 3, 2)

        if self.use_l1:
            gram_losses = torch.sqrt((fp_gram - tp_gram).square().flatten(1, -1).mean(dim=1) + 1).mean(dim=0)
        else:
            gram_losses = (fp_gram - tp_gram).square().flatten(1, -1).mean(dim=1).mean(dim=0)
        return clsloss, losses, gram_losses

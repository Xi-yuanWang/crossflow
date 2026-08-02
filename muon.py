import torch
from torch import nn
from torch.optim import Muon, AdamW

class MuonAdamWOptimizer:
    def __init__(self, model, lr=1e-4, weight_decay=0.01,
                 muon_kwargs=None, adamw_kwargs=None, low_momentum_for_discriminator=False):
        muon_kwargs = muon_kwargs or {}
        adamw_kwargs = adamw_kwargs or {}

        muon_params = []
        adamw_params = []

        muon_discriminator_params = []
        adamw_discriminator_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2:
                if "discriminator" in name:
                    muon_discriminator_params.append(param)
                else:
                    muon_params.append(param)
            else:
                if "discriminator" in name:
                    adamw_discriminator_params.append(param)
                else:
                    adamw_params.append(param)

        self.muon_optim = Muon(
            [{"params": muon_params}, {"params": muon_discriminator_params, "momentum": 0.5 if low_momentum_for_discriminator else 0.95, "weight_decay": 1e-1 if low_momentum_for_discriminator else weight_decay}],
            lr=lr,
            weight_decay=weight_decay,
            adjust_lr_fn="match_rms_adamw", 
            **muon_kwargs
        )

        self.adamw_optim = AdamW(
            [{"params": adamw_params}, {"params": adamw_discriminator_params, "betas": (0.5, 0.9) if low_momentum_for_discriminator else (0.9, 0.95), "weight_decay": 1e-1 if low_momentum_for_discriminator else weight_decay}],
            lr=lr,
            weight_decay=weight_decay,
            **adamw_kwargs
        )

    def zero_grad(self):
        if self.muon_optim is not None:
            self.muon_optim.zero_grad()
        if self.adamw_optim is not None:
            self.adamw_optim.zero_grad()

    def step(self):
        if self.muon_optim is not None:
            self.muon_optim.step()
        if self.adamw_optim is not None:
            self.adamw_optim.step()

    @property
    def param_groups(self):
        groups = []
        if self.muon_optim is not None:
            groups.extend(self.muon_optim.param_groups)
        if self.adamw_optim is not None:
            groups.extend(self.adamw_optim.param_groups)
        return groups

    def state_dict(self):
        return {
            "muon_optim": self.muon_optim.state_dict() if self.muon_optim is not None else None,
            "adamw_optim": self.adamw_optim.state_dict() if self.adamw_optim is not None else None,
        }

    def load_state_dict(self, state_dict):
        muon_state = state_dict.get("muon_optim")
        adamw_state = state_dict.get("adamw_optim")

        if self.muon_optim is not None and muon_state is not None:
            self.muon_optim.load_state_dict(muon_state)
        if self.adamw_optim is not None and adamw_state is not None:
            self.adamw_optim.load_state_dict(adamw_state)
import torch
import torch.nn as nn
from lpips import LPIPS
from kornia import color
# from taming.modules.losses.vqperceptual import *

class ImageSecretLoss(nn.Module):
    def __init__(self, recon_type='rgb', recon_weight=1., perceptual_weight=1.0, secret_weight=10., kl_weight=0.000001, logvar_init=0.0) -> None:
        super().__init__()
        self.recon_type = recon_type
        assert recon_type in ['rgb', 'yuv']
        if recon_type == 'yuv':
            self.register_buffer('yuv_scales', torch.tensor([1,100,100]).unsqueeze(1).float())  # [3,1]
        self.recon_weight = recon_weight
        self.perceptual_weight = perceptual_weight
        self.secret_weight = secret_weight
        self.kl_weight = kl_weight
        self.perceptual_loss = LPIPS().eval()
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
    
    def compute_recon_loss(self, inputs, reconstructions):
        if self.recon_type == 'rgb':
            rec_loss = torch.abs(inputs - reconstructions).mean(dim=[1,2,3])
        elif self.recon_type == 'yuv':
            reconstructions_yuv = color.rgb_to_yuv((reconstructions + 1) / 2)
            inputs_yuv = color.rgb_to_yuv((inputs + 1) / 2)
            yuv_loss = torch.mean((reconstructions_yuv - inputs_yuv)**2, dim=[2,3])
            rec_loss = torch.mm(yuv_loss, self.yuv_scales).squeeze(1)
        else:
            raise ValueError(f"Unknown recon type {self.recon_type}")
        return rec_loss
    
    def forward(self, inputs, reconstructions, posteriors, secret_gt, secret_pred, global_step):
        loss_dict = {}
        rec_loss = self.compute_recon_loss(inputs.contiguous(), reconstructions.contiguous())

        loss = rec_loss*self.recon_weight
        loss_dict['rec_loss'] = rec_loss.mean()

        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous()).mean(dim=[1,2,3])
            loss += self.perceptual_weight * p_loss
            loss_dict['p_loss'] = p_loss.mean()

        loss = loss / torch.exp(self.logvar) + self.logvar
        if self.kl_weight > 0:
            kl_loss = posteriors.kl()
            loss += kl_loss*self.kl_weight
            loss_dict['kl_loss'] = kl_loss.mean()

        secret_loss = self.bce(secret_pred, secret_gt).mean(dim=1)
        loss = (loss +secret_loss*self.secret_weight) / (1+self.secret_weight)
        loss_dict['secret_loss'] = secret_loss.mean()

        bit_acc = ((secret_pred.detach() > 0).float() == secret_gt).float().mean()
        loss_dict['bit_acc'] = bit_acc

        loss_dict['loss'] = loss.mean()

        return loss.mean(), loss_dict



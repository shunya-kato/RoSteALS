import numpy as np
import einops
import torch
import torch as th
import torch.nn as nn
from torch.nn import functional as thf
import pytorch_lightning as pl
import torchvision
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
from contextlib import contextmanager, nullcontext
from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config, default
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
import lpips
from kornia import color

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class SecretEncoder(nn.Module):
    def __init__(self, secret_len=100, dims=2, ch=320, resolution=64) -> None:
        super().__init__()
        log_resolution = int(np.log2(resolution))
        self.model = nn.Sequential(
            nn.Linear(secret_len, 16*16*ch),
            nn.SiLU(),
            View(-1, ch, 16, 16),
            nn.Upsample(scale_factor=(2**(log_resolution-4), 2**(log_resolution-4))),
            # nn.Linear(secret_len, 16*16*ch),nn.SiLU(), 
            # nn.Linear(16*16*ch, 64*64*ch),nn.SiLU(), 
            # View(-1, ch, 64, 64),
            # conv_nd(dims, ch, ch, 3, padding=1),
            # nn.SiLU(),
            # conv_nd(dims, 64, 256, 3, padding=1),
            # nn.SiLU(),
            zero_module(conv_nd(dims, ch, ch, 3, padding=1))
        )
    def forward(self, x, c):
        c = self.model(c)
        return x + c


class ControlAE(pl.LightningModule):
    def __init__(self,
                 first_stage_key,
                 first_stage_config,
                 control_key,
                 control_config,
                 decoder_config,
                 use_ema=False,
                 scale_factor=1.,
                 ckpt_path="__none__",
                 ):
        super().__init__()
        self.scale_factor = scale_factor
        self.control_key = control_key
        self.first_stage_key = first_stage_key
        self.ae = instantiate_from_config(first_stage_config)
        # freeze first stage
        self.ae.eval()
        self.ae.train = disabled_train
        for p in self.ae.parameters():
            p.requires_grad = False
        self.control = instantiate_from_config(control_config)
        self.decoder = instantiate_from_config(decoder_config)

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.lpips_loss = lpips.LPIPS(net="alex", verbose=False)
        self.register_buffer('yuv_scales', torch.tensor([1,100,100]).unsqueeze(1).float())  # [3,1]

        # early training phase
        self.fixed_input = True
        self.fixed_x = None
        self.fixed_img = None

        self.use_ema = use_ema
        if self.use_ema:
            print('Using EMA')
            self.control_ema = LitEma(self.control)
            self.decoder_ema = LitEma(self.decoder)
            print(f"Keeping EMAs of {len(list(self.control_ema.buffers()) + list(self.decoder_ema.buffers()))}.")

        if ckpt_path != '__none__':
            self.init_from_ckpt(ckpt_path, ignore_keys=[])

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.control_ema.store(self.control.parameters())
            self.decoder_ema.store(self.decoder.parameters())
            self.control_ema.copy_to(self.control)
            self.decoder_ema.copy_to(self.decoder)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.control_ema.restore(self.control.parameters())
                self.decoder_ema.restore(self.decoder.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.control_ema(self.control)
            self.decoder_ema(self.decoder)

    def compute_loss(self, pred, target):
        # return thf.mse_loss(pred, target, reduction="none").mean(dim=(1, 2, 3))
        lpips_loss = self.lpips_loss(pred, target).mean(dim=[1,2,3])
        pred_yuv = color.rgb_to_yuv((pred + 1) / 2)
        target_yuv = color.rgb_to_yuv((target + 1) / 2)
        yuv_loss = torch.mean((pred_yuv - target_yuv)**2, dim=[2,3])
        yuv_loss = 1.5*torch.mm(yuv_loss, self.yuv_scales).squeeze(1)
        return lpips_loss + yuv_loss

    def forward(self, x, c):
        return self.control(x, c)

    @torch.no_grad()
    def get_input(self, batch, return_first_stage=False, bs=None):
        image = batch[self.first_stage_key]
        control = batch[self.control_key]
        if bs is not None:
            image = image[:bs]
            control = control[:bs]
        # encode image 1st stage
        image = einops.rearrange(image, "b h w c -> b c h w").contiguous()
        x = self.encode_first_stage(image).detach()
        out = [x, control]
        if return_first_stage:
            image_rec = self.decode_first_stage(x)
            out.extend([image, image_rec])
        return out

    def decode_first_stage(self, z):
        z = 1./self.scale_factor * z
        image_rec = self.ae.decode(z)
        return image_rec
    
    def encode_first_stage(self, image):
        encoder_posterior = self.ae.encode(image)
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def shared_step(self, batch):
        x, c, img, _ = self.get_input(batch, return_first_stage=True)
        if self.training and self.fixed_input:
            if self.fixed_x is None:
                print('Using fixed input first!')
                self.fixed_x = x.detach()
                self.fixed_img = img.detach()
            x = self.fixed_x
            img = self.fixed_img
        x = self.control(x, c)
        image_rec = self.decode_first_stage(x)
        # resize
        if img.shape[-1] > 256:
            img =  thf.interpolate(img, size=(256, 256), mode='bilinear', align_corners=False).detach()
            image_rec =  thf.interpolate(image_rec, size=(256, 256), mode='bilinear', align_corners=False)
        pred = self.decoder(image_rec)

        loss_dict = {}
        loss_image = self.compute_loss(image_rec, img)
        loss_dict["loss_image"] = loss_image.mean()
        loss_control = self.bce(pred, c).mean(dim=1)
        loss_dict["loss_control"] = loss_control.mean()
        loss_weight = 10.
        loss = (loss_image + loss_weight * loss_control).mean() / (1 + loss_weight)
        loss_dict["loss"] = loss 
        bit_acc = ((pred.detach() > 0).float() == c).float().mean()
        loss_dict["bit_acc"] = bit_acc
        if bit_acc.item() > 0.9:
            self.fixed_input = False
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)
        loss_dict = {f"train/{key}": val for key, val in loss_dict.items()}
        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)
        
        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)
        # if self.use_scheduler:
        #     lr = self.optimizers().param_groups[0]['lr']
        #     self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        loss_dict_no_ema = {f"val/{key}": val for key, val in loss_dict_no_ema.items()}
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {'val/' + key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
    
    @torch.no_grad()
    def log_images(self, batch, N=4, **kwargs):
        log = dict()
        x, c, img, img_recon = self.get_input(batch, return_first_stage=True, bs=N)
        x = self.control(x, c)
        image_out = self.decode_first_stage(x)
        log['image'] = img
        log['image_out'] = image_out
        log['image_recon_org'] = img_recon
        return log
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.AdamW(params, lr=lr)
        return optimizer
    





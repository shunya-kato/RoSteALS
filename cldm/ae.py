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
from ldm.modules.diffusionmodules.model import Encoder
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


class SecretEncoder3(nn.Module):
    def __init__(self, secret_len, base_res=16, resolution=64) -> None:
        super().__init__()
        log_resolution = int(np.log2(resolution))
        log_base = int(np.log2(base_res))
        self.secret_len = secret_len
        self.secret_scaler = nn.Sequential(
            nn.Linear(secret_len, base_res*base_res*3),
            nn.SiLU(),
            View(-1, 3, base_res, base_res),
            nn.Upsample(scale_factor=(2**(log_resolution-log_base), 2**(log_resolution-log_base))),  # chx16x16 -> chx256x256
            zero_module(conv_nd(2, 3, 3, 3, padding=1))
        )  # secret len -> ch x res x res
    
    def copy_encoder_weight(self, ae_model):
        # misses, ignores = self.load_state_dict(ae_state_dict, strict=False)
        return None

    def encode(self, x):
        x = self.secret_scaler(x)
        return x
    
    def forward(self, x, c):
        # x: [B, C, H, W], c: [B, secret_len]
        c = self.encode(c)
        return c, None

class SecretEncoder2(nn.Module):
    def __init__(self, secret_len, embed_dim, ddconfig, ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ema_decay=None,
                 learn_logvar=False) -> None:
        super().__init__()
        log_resolution = int(np.log2(ddconfig.resolution))
        self.secret_len = secret_len
        self.learn_logvar = learn_logvar
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.encoder.conv_out = zero_module(self.encoder.conv_out)
        self.embed_dim = embed_dim

        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))

        if monitor is not None:
            self.monitor = monitor

        self.secret_scaler = nn.Sequential(
            nn.Linear(secret_len, 32*32*ddconfig.out_ch),
            nn.SiLU(),
            View(-1, ddconfig.out_ch, 32, 32),
            nn.Upsample(scale_factor=(2**(log_resolution-5), 2**(log_resolution-5))),  # chx16x16 -> chx256x256
            # zero_module(conv_nd(2, ddconfig.out_ch, ddconfig.out_ch, 3, padding=1))
        )  # secret len -> ch x res x res
        # out_resolution = ddconfig.resolution//(len(ddconfig.ch_mult)-1)
        # self.out_layer = zero_module(conv_nd(2, ddconfig.out_ch, ddconfig.out_ch, 3, padding=1))

        self.use_ema = ema_decay is not None
        if self.use_ema:
            self.ema_decay = ema_decay
            assert 0. < ema_decay < 1.
            self.model_ema = LitEma(self, decay=ema_decay)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)


    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        misses, ignores = self.load_state_dict(sd, strict=False)
        print(f"[SecretEncoder] Restored from {path}, misses: {misses}, ignores: {ignores}")

    def copy_encoder_weight(self, ae_model):
        # misses, ignores = self.load_state_dict(ae_state_dict, strict=False)
        return None
        self.encoder.load_state_dict(ae_model.encoder.state_dict())
        self.quant_conv.load_state_dict(ae_model.quant_conv.state_dict())

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")
    
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        h = self.encoder(x)
        posterior = h
        return posterior
    
    def forward(self, x, c):
        # x: [B, C, H, W], c: [B, secret_len]
        c = self.secret_scaler(c)
        x = torch.cat([x, c], dim=1)
        z = self.encode(x)
        # z = self.out_layer(z)
        return z, None

class SecretEncoder(nn.Module):
    def __init__(self, secret_len, embed_dim, ddconfig, ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ema_decay=None,
                 learn_logvar=False) -> None:
        super().__init__()
        log_resolution = int(np.log2(ddconfig.resolution))
        self.secret_len = secret_len
        self.learn_logvar = learn_logvar
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.embed_dim = embed_dim

        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))

        if monitor is not None:
            self.monitor = monitor

        self.use_ema = ema_decay is not None
        if self.use_ema:
            self.ema_decay = ema_decay
            assert 0. < ema_decay < 1.
            self.model_ema = LitEma(self, decay=ema_decay)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.secret_scaler = nn.Sequential(
            nn.Linear(secret_len, 32*32*ddconfig.out_ch),
            nn.SiLU(),
            View(-1, ddconfig.out_ch, 32, 32),
            nn.Upsample(scale_factor=(2**(log_resolution-5), 2**(log_resolution-5))),  # chx16x16 -> chx256x256
            zero_module(conv_nd(2, ddconfig.out_ch, ddconfig.out_ch, 3, padding=1))
        )  # secret len -> ch x res x res
        # out_resolution = ddconfig.resolution//(len(ddconfig.ch_mult)-1)
        self.out_layer = zero_module(conv_nd(2, ddconfig.out_ch, ddconfig.out_ch, 3, padding=1))

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        misses, ignores = self.load_state_dict(sd, strict=False)
        print(f"[SecretEncoder] Restored from {path}, misses: {misses}, ignores: {ignores}")

    def copy_encoder_weight(self, ae_model):
        # misses, ignores = self.load_state_dict(ae_state_dict, strict=False)
        self.encoder.load_state_dict(ae_model.encoder.state_dict())
        self.quant_conv.load_state_dict(ae_model.quant_conv.state_dict())

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")
    
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior
    
    def forward(self, x, c):
        # x: [B, C, H, W], c: [B, secret_len]
        c = self.secret_scaler(c)
        x = x + c
        posterior = self.encode(x)
        z = posterior.sample()
        z = self.out_layer(z)
        return z, posterior


class ControlAE(pl.LightningModule):
    def __init__(self,
                 first_stage_key,
                 first_stage_config,
                 control_key,
                 control_config,
                 decoder_config,
                 loss_config,
                 use_ema=False,
                 scale_factor=1.,
                 ckpt_path="__none__",
                 ):
        super().__init__()
        self.scale_factor = scale_factor
        self.control_key = control_key
        self.first_stage_key = first_stage_key
        self.ae = instantiate_from_config(first_stage_config)
        self.control = instantiate_from_config(control_config)
        self.decoder = instantiate_from_config(decoder_config)

        # copy weights from first stage
        self.control.copy_encoder_weight(self.ae)
        # freeze first stage
        self.ae.eval()
        self.ae.train = disabled_train
        for p in self.ae.parameters():
            p.requires_grad = False

        # self.bce = nn.BCEWithLogitsLoss(reduction="none")
        # self.lpips_loss = lpips.LPIPS(net="alex", verbose=False)
        # self.register_buffer('yuv_scales', torch.tensor([1,100,100]).unsqueeze(1).float())  # [3,1]
        self.loss_layer = instantiate_from_config(loss_config)

        # early training phase
        self.fixed_input = True
        self.fixed_x = None
        self.fixed_img = None
        self.fixed_input_recon = None

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

    def forward(self, x, image, c):
        eps, posterior = self.control(image, c)
        return x + eps, posterior

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
        image_rec = self.decode_first_stage(x).detach()
        
        # check if using fixed input (early training phase)
        if self.training and self.fixed_input:
            if self.fixed_x is None:  # first iteration
                print('Warmup training - using fixed input image for now!')
                self.fixed_x = x.detach().clone()
                self.fixed_img = image.detach().clone()
                self.fixed_input_recon = image_rec.detach().clone()
            x = self.fixed_x
            image = self.fixed_img
            image_rec = self.fixed_input_recon
        
        out = [x, control]
        if return_first_stage:
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
        x, posterior = self(x, img, c)
        image_rec = self.decode_first_stage(x)
        # resize
        if img.shape[-1] > 256:
            img =  thf.interpolate(img, size=(256, 256), mode='bilinear', align_corners=False).detach()
            image_rec =  thf.interpolate(image_rec, size=(256, 256), mode='bilinear', align_corners=False)
        pred = self.decoder(image_rec)

        loss, loss_dict = self.loss_layer(img, image_rec, posterior, c, pred, self.global_step)
        bit_acc = loss_dict["bit_acc"]

        # loss_dict = {}
        # loss_image = self.compute_loss(image_rec, img)
        # loss_dict["loss_image"] = loss_image.mean()
        # loss_control = self.bce(pred, c).mean(dim=1)
        # loss_dict["loss_control"] = loss_control.mean()
        # loss_weight = 10.
        # loss = (loss_image + loss_weight * loss_control).mean() / (1 + loss_weight)
        # loss_dict["loss"] = loss 
        # bit_acc = ((pred.detach() > 0).float() == c).float().mean()
        # loss_dict["bit_acc"] = bit_acc
        if (bit_acc.item() > 0.9) and self.fixed_input:
            print('High bit acc achieved. Switch to full image dataset training from now.')
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
        x, _ = self(x, img, c)
        image_out = self.decode_first_stage(x)
        log['image_in'] = img
        log['image_out'] = image_out
        log['image_in_recon'] = img_recon
        return log
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.AdamW(params, lr=lr)
        return optimizer
    





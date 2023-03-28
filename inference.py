
import os, sys, torch 
import argparse
from pathlib import Path
import numpy as np
import pickle
import pytorch_lightning as pl
from torchvision import transforms
import argparse
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image
from time import time
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from tools.eval_metrics import compute_psnr, compute_ssim, compute_mse, compute_lpips, compute_sifid, resize_array, resize_tensor
from tools.augment_imagenetc import RandomImagenetC
import lpips
from tools.sifid import SIFID
from tools.image_dataset import dataset_wrapper
from tools.helpers import welcome_message
from tools.ecc import ECC

def unormalize(x):
    # convert x in range [-1, 1], (B,C,H,W), tensor to [0, 255], uint8, numpy, (B,H,W,C)
    x = torch.clamp((x + 1) * 127.5, 0, 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    return x

def main(args):
    print(welcome_message())
    # Load model
    config = OmegaConf.load(args.config).model
    secret_len = config.params.control_config.params.secret_len
    config.params.decoder_config.params.secret_len = secret_len
    model = instantiate_from_config(config)
    state_dict = torch.load(args.weight, map_location=torch.device('cpu'))
    if 'global_step' in state_dict:
        print(f'Global step: {state_dict["global_step"]}, epoch: {state_dict["epoch"]}')

    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    misses, ignores = model.load_state_dict(state_dict, strict=False)
    print(f'Missed keys: {misses}\nIgnore keys: {ignores}')
    model = model.cuda()
    model.eval()

    # cover
    tform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    cover = Image.open(args.cover).convert('RGB')
    cover = tform(cover).unsqueeze(0).cuda()  # 1, 3, 256, 256

    # secret
    ecc = ECC()
    secret = ecc.encode_text([args.secret])  # 1, 100
    secret = torch.from_numpy(secret).cuda().float()  # 1, 100

    # inference
    with torch.no_grad():
        z = model.encode_first_stage(cover)
        z_embed, _ = model(z, None, secret)
        stego = model.decode_first_stage(z_embed)  # 1, 3, 256, 256
        stego_uint8 = unormalize(stego)[0]  # 256, 256, 3
        secret_pred = (model.decoder(stego) > 0).cpu().numpy()  # 1, 100
        print(f'Bit acc: {np.mean(secret_pred == secret.cpu().numpy())}')
        secret_decoded = ecc.decode_text(secret_pred)[0]
        print(f'Recovered secret: {secret_decoded}')

        # save stego
        Image.fromarray(stego_uint8).save('stego.png')
        print('Stego saved to stego.png')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--config", default='models/VQ4_s100_mir100k2.yaml', help="Path to config file.")
    parser.add_argument('-w', "--weight", default='/mnt/fast/nobackup/scratch4weeks/tb0035/projects/diffsteg/controlnet/VQ4_s100_mir100k2/checkpoints/epoch=000017-step=000449999.ckpt', help="Path to checkpoint file.")
    parser.add_argument(
        "--image_size", type=int, default=256, help="Height and width of square images."
    )
    parser.add_argument(
        "--secret", default='secrets', help="secret message, 7 characters max"
    )
    parser.add_argument(
        "--cover", default='examples/00096.png', help="cover image path"
    )
    args = parser.parse_args()
    main(args)
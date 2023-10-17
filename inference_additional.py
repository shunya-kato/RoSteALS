
import os, sys, torch 
import argparse
from pathlib import Path
import numpy as np
from torchvision import transforms
import argparse
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image
from tools.eval_metrics import compute_psnr, compute_ssim, compute_mse, compute_lpips, compute_sifid
import lpips
from tools.sifid import SIFID
from tools.helpers import welcome_message
from tools.ecc import ECC


# def unormalize(x):
#     # convert x in range [-1, 1], (B,C,H,W), tensor to [0, 255], uint8, numpy, (B,H,W,C)
#     x = torch.clamp((x + 1) * 127.5, 0, 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
#     return x

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
    cover_org = Image.open(args.cover).convert('RGB')
    w,h = cover_org.size
    cover = tform(cover_org).unsqueeze(0).cuda()  # 1, 3, 256, 256

    # secret
    ecc = ECC()
    secret = ecc.encode_text([args.secret])  # 1, 100
    secret = torch.from_numpy(secret).cuda().float()  # 1, 100

    additional_secret = ecc.encode_text([args.additional_secret])  # 1, 100
    additional_secret = torch.from_numpy(additional_secret).cuda().float()  # 1, 100

    # inference
    lpips_alex = lpips.LPIPS(net='alex').cuda()
    sifid_model = SIFID()
    with torch.no_grad():
        z = model.encode_first_stage(cover)
        z_embed, _ = model(z, None, secret)
        z_embed, _ = model.additional_forward(z_embed, None, additional_secret)
        stego = model.decode_first_stage(z_embed)  # 1, 3, 256, 256
        res = stego.clamp(-1,1) - cover  # (1,3,256,256) residual
        res = torch.nn.functional.interpolate(res, (h,w), mode='bilinear')
        res = res.permute(0,2,3,1).cpu().numpy()  # (1,h,w,3)
        stego_uint8 = np.clip(res[0] + np.array(cover_org)/127.5-1., -1,1)*127.5+127.5  
        stego_uint8 = stego_uint8.astype(np.uint8)  # (h,w, 3), ndarray, uint8

        # quality metrics
        print(f'Quality metrics at resolution: {h}x{w} (HxW)')
        print(f'MSE: {compute_mse(np.array(cover_org)[None,...], stego_uint8[None,...])}')
        print(f'PSNR: {compute_psnr(np.array(cover_org)[None,...], stego_uint8[None,...])}')
        print(f'SSIM: {compute_ssim(np.array(cover_org)[None,...], stego_uint8[None,...])}')
        cover_org_norm = torch.from_numpy(np.array(cover_org)[None,...]/127.5-1.).permute(0,3,1,2).float().cuda()
        stego_norm = torch.from_numpy(stego_uint8[None,...]/127.5-1.).permute(0,3,1,2).float().cuda()
        print(f'LPIPS: {compute_lpips(cover_org_norm, stego_norm, lpips_alex)}')
        print(f'SIFID: {compute_sifid(cover_org_norm, stego_norm, sifid_model)}')

        # decode secret
        print('Extracting secret...')
        secret_pred = (model.decoder(stego) > 0).cpu().numpy()  # 1, 100
        print(f'Bit acc: {np.mean(secret_pred == secret.cpu().numpy())}')
        secret_decoded = ecc.decode_text(secret_pred)[0]
        print(f'Recovered secret: {secret_decoded}')

        # save stego
        Image.fromarray(stego_uint8).save(args.output)
        print(f'Stego saved to {args.output}')

        # decode additional secret
        print('Extracting secret...')
        stego = torch.nn.functional.interpolate(stego, (64,64), mode='bilinear', align_corners=False)
        stego = torch.nn.functional.interpolate(stego, (256,256), mode='bilinear', align_corners=False)
        additional_secret_pred = (model.decoder(stego) > 0).cpu().numpy()  # 1, 100
        print(f'Bit acc: {np.mean(additional_secret_pred == additional_secret.cpu().numpy())}')
        additional_secret_decoded = ecc.decode_text(additional_secret_pred)[0]
        print(f'Recovered additional secret: {additional_secret_decoded}')


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
        "--additional_secret", default='sonycsl', help="secret message, 7 characters max"
    )
    parser.add_argument(
        "--cover", default='examples/00096.png', help="cover image path"
    )
    parser.add_argument(
        "-o", "--output", default='stego.png', help="output stego image path"
    )
    args = parser.parse_args()
    main(args)
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
streamlit app demo
how to run:
streamlit run app.py --server.port 8501

@author: Tu Bui @surrey.ac.uk
"""
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
from tools.augment_imagenetc import RandomImagenetC
from io import BytesIO
from tools.helpers import welcome_message
from tools.ecc import BCH, RSC

import streamlit as st

# noise = RandomImagenetC(phase='test')
# corrupt_methods = [noise.method_names[i] for i in noise.corrupt_ids]
model_names = ['RoSteALS', 'RivaGAN']
SECRET_LEN = 160

def unormalize(x):
    # convert x in range [-1, 1], (B,C,H,W), tensor to [0, 255], uint8, numpy, (B,H,W,C)
    x = torch.clamp((x + 1) * 127.5, 0, 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    return x

def to_bytes(x):
    x = Image.fromarray(x)
    buf = BytesIO()
    x.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

def load_RoSteALS():
    config_file = '/mnt/fast/nobackup/scratch4weeks/tb0035/projects/diffsteg/controlnet/VQ4_s160_full_lw2/configs/-project.yaml'
    weight_file = '/mnt/fast/nobackup/scratch4weeks/tb0035/projects/diffsteg/controlnet/VQ4_s160_full_lw2/checkpoints/epoch=000002-step=000399999.ckpt'
    config = OmegaConf.load(config_file).model
    secret_len = config.params.control_config.params.secret_len
    assert SECRET_LEN == secret_len
    config.params.decoder_config.params.secret_len = secret_len
    model = instantiate_from_config(config)
    state_dict = torch.load(weight_file, map_location=torch.device('cpu'))
    if 'global_step' in state_dict:
        print(f'Global step: {state_dict["global_step"]}, epoch: {state_dict["epoch"]}')

    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    misses, ignores = model.load_state_dict(state_dict, strict=False)
    print(f'Missed keys: {misses}\nIgnore keys: {ignores}')
    model = model.cuda()
    model.eval()
    return model

def embed_secret(model_name, model, im, secret):
    if model_name == 'RoSteALS':
        with torch.no_grad():
            z = model.encode_first_stage(im)
            z_embed, _ = model(z, None, secret)
            stego = model.decode_first_stage(z_embed)  # 1, 3, 256, 256
            stego_uint8 = unormalize(stego)[0]  # (256, 256, 3), ndarray, uint8
    else:
        raise NotImplementedError
    return stego_uint8

def decode_secret(model_name, model, im):
    if model_name == 'RoSteALS':
        with torch.no_grad():
            secret_pred = (model.decoder(im) > 0).cpu().numpy()  # 1, 100
    else:
        raise NotImplementedError
    return secret_pred

@st.cache_resource
def load_model(model_name):
    if model_name == 'RoSteALS':
        prep = transforms.Resize((256,256))  # preprocess step for display purpose
        tform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        model = load_RoSteALS()
    else:
        raise NotImplementedError
    return model, prep, tform

def app():
    st.title('Watermarking Demo')
    # setup model
    model_name = st.selectbox("Choose the model", model_names)
    model, prep, tform = load_model(model_name)
    
    # ecc
    # ecc = RSC(data_bytes=16, ecc_bytes=4, verbose=True)  # 20 bytes in total with Reed Solomon code
    ecc = BCH(285, 10, SECRET_LEN, verbose=True)
    assert ecc.get_total_len() == SECRET_LEN

    # setup st
    st.subheader("Input")
    image_file = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])
    if image_file is not None:
        im = Image.open(image_file).convert('RGB')
        st.image(im, width=256)
        im = prep(im)
    secret_text = st.text_input(f'Input the secret (max {ecc.data_len} chars)', 'My secrets')
    assert len(secret_text) <= ecc.data_len

    # embed
    st.subheader("Embed results")
    status = st.empty()
    if image_file is not None and secret_text is not None:
        secret = ecc.encode_text([secret_text])  # (1, len)
        secret = torch.from_numpy(secret).float().cuda()
        im = tform(im).unsqueeze(0).cuda()  # (1,3,H,W)
        stego = embed_secret(model_name, model, im, secret)
        st.image(stego)

        # download button
        stego_bytes = to_bytes(stego)
        st.download_button(label='Download image', data=stego_bytes, file_name='stego.png', mime='image/png')

        # verify secret
        stego_processed = tform(Image.fromarray(stego)).unsqueeze(0).cuda()
        secret_pred = decode_secret(model_name, model, stego_processed)
        bit_acc = (secret_pred == secret.cpu().numpy()).mean()
        secret_pred = ecc.decode_text(secret_pred)[0]
        status.markdown('**Secret recovery check:** ' + secret_pred, unsafe_allow_html=True)
        status.markdown('**Bit accuracy:** ' + str(bit_acc), unsafe_allow_html=True)

if __name__ == '__main__':
    app()
    

    
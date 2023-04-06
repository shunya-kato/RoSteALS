# RoSteALS

Official implementation of [RoSteALS: Robust Steganography using Autoencoder Latent Space]().

### Environment

We tested with pytorch 1.11, torchvision 0.12 and cuda 11.3, but other pytorch version probably works, too. To reproduce the environment, please check [dependencies](dependencies).

# Training
## Data Preparation
TODO: instructions to download and prepare the MIRFlickR dataset.

Update the data path in the config file at [models/VQ4_mir.yaml](models/VQ4_mir.yaml).

## Train
```
python train.py --config models/VQ4_mir.yaml --secret_len 100 --max_image_weight_ratio 10 --batch_size 4 -o saved_models

```
where batch_size=4 is enough to fit a 24GB GPU.

# Inference
TODO: upload trained model, inference demo

# Acknowledgement
The code is inspired from [Stable Diffusion](https://github.com/CompVis/stable-diffusion) and [ControlNet](https://github.com/lllyasviel/ControlNet). 


# Citation
TODO: update

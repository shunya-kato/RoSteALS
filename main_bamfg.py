#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pip install open_clip_torch==2.0.2
@author: Tu Bui @University of Surrey
"""
import os
from cldm.hack import disable_verbosity, enable_sliced_attention
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pathlib import Path
from cldm.model import create_model, load_state_dict
from ldm.util import instantiate_from_config
import argparse


def get_parser():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/mnt/fast/nobackup/users/tb0035/projects/diffsteg/ControlNet/models/bamfg.yaml')
    parser.add_argument('--init_ckpt', type=str, default='/mnt/fast/nobackup/users/tb0035/projects/diffsteg/ControlNet/models/control_sd15_ini.ckpt')
    parser.add_argument('-o', '--output', type=str, default='/mnt/fast/nobackup/scratch4weeks/tb0035/projects/diffsteg/controlnet/bamfg')
    parser.add_argument('--sd_locked', type=str2bool, default=True)
    parser.add_argument('--only_mid_control', action='store_true')
    parser.add_argument('--resume', type=str2bool, default=True)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size, 8 for 1 A100 80GB GPU')
    return parser.parse_args()

def app(args):
    # Configs
    config_path = args.config
    resume_path = args.init_ckpt
    output = args.output
    sd_locked = args.sd_locked
    only_mid_control = args.only_mid_control

    ckptdir = os.path.join(output, 'checkpoints')
    cfgdir = os.path.join(output, 'configs')
    resumedir = output if os.path.exists(os.path.join(ckptdir, 'last.ckpt')) else ''

    config = OmegaConf.load(config_path)
    # data
    data_config = config.pop("data", OmegaConf.create())
    data_config.params.batch_size = args.batch_size
    data = instantiate_from_config(data_config)
    data.prepare_data()
    data.setup()
    print("#### Data #####")
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    # callbacks
    pl_config = config.pop("lightning", OmegaConf.create())
    image_logger_callback = pl_config.callbacks.get("image_logger", OmegaConf.create())
    callbacks = [
        dict(target='cldm.logger.SetupCallback', 
        params={'resume': resumedir, 'now': '', 'logdir': output, 'ckptdir': ckptdir, 'cfgdir': cfgdir, 'config': config, 'lightning_config': pl_config}),

        dict(target='pytorch_lightning.callbacks.ProgressBar', 
        params={'refresh_rate': 50}),

        image_logger_callback        
    ]
    callbacks = [instantiate_from_config(c) for c in callbacks]

    
    # logger
    logger = dict(target='pytorch_lightning.loggers.TestTubeLogger', params={'name': 'testtube', 'save_dir': output})
    logger = instantiate_from_config(logger)

    # trainer
    trainer_kwargs = dict(gpus=args.gpus, precision=32, callbacks=callbacks, logger=logger)
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.logdir = output

    # model
    model = instantiate_from_config(config.model).cpu()
    loaded_state_dict = load_state_dict(resume_path, location='cpu')
    # current_model_dict = model.state_dict()
    # new_state_dict={k:v if v.size()==current_model_dict[k].size()  else  current_model_dict[k] for k,v in zip(current_model_dict.keys(), loaded_state_dict.values())}
    # model.load_state_dict(new_state_dict, strict=False)
    model.load_state_dict(loaded_state_dict)

    # model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
    model.learning_rate = pl_config.trainer.base_learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # Train!
    model.model.diffusion_model.input_blocks.requires_grad_(False)
    trainer.fit(model, data)

if __name__ == "__main__":
    disable_verbosity()
    args = get_parser()
    app(args)
    

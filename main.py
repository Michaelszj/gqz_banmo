# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from absl import app
from absl import flags
import cv2
import os.path as osp
import sys
sys.path.insert(0,'third_party')
import pdb
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import argparse
from omegaconf import OmegaConf
from nnutils.train_utils import v2s_trainer

opts = flags.FLAGS

def main(_):
    torch.cuda.set_device(opts.local_rank)
    torch.manual_seed(0)
    torch.cuda.manual_seed(1)
    torch.manual_seed(0)
    
    
    # override default config from cli
    extra_opt = OmegaConf.load(opts.config)
    trainer = v2s_trainer(opts)
    data_info = trainer.init_dataset()  
    trainer.define_model(data_info,extra_opt) 
    trainer.init_training()
    # import pdb;pdb.set_trace()
    trainer.train()

if __name__ == '__main__':
    app.run(main)

import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models

import warnings
warnings.filterwarnings("ignore")

from utils import *
from config import configs
from trainer import FinetuneTrainer

class FullFTNet(nn.Module):
    def __init__(self, lr, model=None, linear_eval=False, use_nn=True, num_classes=1000, feat_dim=2048):
        super().__init__()
        self.lr = lr
        self.linear_eval = linear_eval
        if self.linear_eval:
          model.eval()
          model.requires_grad_(False)
        
        self.model = model
        
        if use_nn:
            print("Number of classes", num_classes)
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(feat_dim,num_classes)
            )
            self.model = torch.nn.Sequential(
                model, self.mlp
            )
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, X):
        return self.model(X)          
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config', type=str, default='base', choices=configs.keys(), help='Configuration to use')
    parser.add_argument('-d','--data', type=str, default='tinyimgenet', choices=['tinyimagenet', 'caltech', 'voc2007'], help='Finetune Dataset to use')
    parser.add_argument('-t','--train', action='store_true', help='train the model')


    args = parser.parse_args()

    config = configs[args.config](args.data)
    reproducibility(config)
    
    backbone = models.resnet50(pretrained=False)
    backbone.fc = nn.Identity()

    if not config.random and os.path.exists(config.ckpt):
        print('Loading pretrained weights from ', config.ckpt)
        checkpoint = torch.load(config.ckpt)
        
        # INITIAL simclr weights
        if 'model_state_dict' in checkpoint.keys():
            backbone.load_state_dict(checkpoint['model_state_dict'])
        
        else:
        # EMILIO's weights
            try:
                backbone.load_state_dict(checkpoint)    
        # KARIMA's weights
            except:
                updated = {k.replace("backbone.", ""):v for k,v in checkpoint.items()}
                backbone.load_state_dict(updated, strict=False)
        
    model = FullFTNet(config.lr, model=backbone, linear_eval=config.linear_eval, use_nn=True, num_classes=config.num_classes)

    trainer = FinetuneTrainer(config, model)

    if args.train:
        trainer.train()

    trainer.validate()

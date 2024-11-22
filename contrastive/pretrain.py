import numpy as np
import os
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from tqdm import tqdm
from torch.optim import SGD, Adam
from torchtune.training.metric_logging import WandBLogger
import warnings
warnings.filterwarnings("ignore")

from utils import *
from config import configs
from trainer import Trainer
import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config', type=str, default='base', choices=configs.keys(), help='Configuration to use')
    parser.add_argument('-t','--train', action="store_true",  help='True to train, False to save weights only')

    args = parser.parse_args()
    config = configs[args.config]()
    print('Loaded configuration for', args.config)
    
    # set seed
    reproducibility(config)

    if config.architecture == 'resnet':
        print('Using ResNet50 backbone')
        backbone = models.resnet50(pretrained=False)
        feat_dim = 2048
    else:
        print('Using ViT backbone')
        backbone = models.vit_b_16(pretrained=False)
        feat_dim = 768
        
    if config.continue_task:
        print('Loading previous task backbone weights from', os.path.basename(config.previous_task_backbone))
        backbone.load_state_dict(torch.load(config.previous_task_backbone), strict=False)

    model = SimCLR_pl(config, model=backbone, feat_dim=feat_dim)

    pre_trainer = Trainer(config, model)

    if args.train:
        print('Training model..')
        pre_trainer.train()

    print(f'Loading backbone weights from {pre_trainer.best_checkpoint}..')
    best_pretrain = weights_update(model, pre_trainer.best_checkpoint)

    resnet_backbone_weights = best_pretrain.model.backbone
    torch.save(resnet_backbone_weights.state_dict(), f'{config.save_prefix}_{config.df}_backbone_weights.ckpt')
    # torch.save({
    #             'model_state_dict': resnet_backbone_weights.state_dict(),
    #             }, f'{config.save_prefix}_{config.df}_backbone_weights.ckpt')

    print('Checkpoint saved!')

if __name__=="__main__":
    main()
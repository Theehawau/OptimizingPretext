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
import wandb



class Trainer():
    def __init__(self, config, model):
        self.config = config
        self.config.batch_size = self.config.batch_size #* torch.cuda.device_count()
        self.model = model 
        self.exp_name = self.config.exp_prefix + self.config.backbone + self.config.df 
        self.loss = ContrastiveLoss(config.batch_size, temperature=self.config.temperature)
        self.optimizer, self.lr_scheduler = self.configure_optimizers()
        self.best_loss = np.inf
        self.epoch = 0
        self.device = config.device
        self.best_checkpoint = f"{self.config.save}/{self.exp_name}/best.ckpt"
        self.global_step = 0
        
        os.makedirs(f"{self.config.save}/{self.exp_name}", exist_ok=True)
        
        # print(self.model)
        self.model = self.model.to(self.device)
        
        if self.config.resume_from_checkpoint:
            self.resume_checkpoint()
        
        print(f'Number of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)/1e6:.2f}M')
        print(f'Number of total parameters: {sum(p.numel() for p in self.model.parameters())/1e6:.2f}')
        
        self.train_loader = self.get_dataloaders()
        
        self.logger = WandBLogger(project='OptimizingPretext', name=self.exp_name, config=config.__dict__)
    

    def train(self):

        for epoch in range(self.epoch, self.config.epochs):
            torch.cuda.empty_cache()
            epoch_loss = 0
            self.model.train()
            tq_obj = tqdm(self.train_loader, desc=f'Epoch {epoch}', total=len(self.train_loader), leave=False)
            acc_loss = 0
            for i , batch in enumerate(tq_obj):
                self.optimizer.zero_grad()
                loss = self.training_step(batch, i).mean()
                tq_obj.set_postfix({'loss': loss.item()})
                acc_loss += loss
                epoch_loss += loss.item()
                if self.global_step % self.config.gradient_accumulation_steps == 0:
                    acc_loss.backward()
                    self.optimizer.step()
                    acc_loss = 0
                
                if self.global_step % 100 == 0:
                    self.logger.log_dict({'Contrastive loss': loss.item(), 
                                          'epoch': epoch}, self.global_step)
                self.global_step+=1
            self.logger.log('Contrastive loss epoch', epoch_loss/len(self.train_loader),
                             self.global_step )
            if epoch_loss < self.best_loss:
                print(f"Saved best ckpt at epoch {epoch}, with loss {epoch_loss/len(self.train_loader)}")
                self.save_best()
                self.best_loss = epoch_loss
            self.epoch += 1
            self.save()
    
    def save(self, path=None):
        to_save = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_loss': self.best_loss,
        }
        torch.save(to_save, path or f"{self.config.save}/{self.exp_name}/last.ckpt")
    
    def save_best(self):
        self.save(path=self.best_checkpoint)
    
    def resume_checkpoint(self):
        checkpoint = torch.load(f"{self.config.save}/{self.exp_name}/last.ckpt")
        self.model.load_state_dict({k.replace('module.',''): v for k, v in checkpoint['state_dict'].items()})
        self.model.to(self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print("Loaded existing checkpoint from", f"{self.config.save}/{self.exp_name}/last.ckpt")
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint.get('global_step', 0)
        self.best_loss = checkpoint.get('best_loss', np.inf)

    def get_dataloaders(self):
        transform = Augment(self.config.img_size)
        df = load_dataset(self.config.dataset_path)
        if self.config.reduce < 1.0:
            df = reduce_dataset(df, self.config.reduce)
        return get_imagenet_dataloader(self.config.batch_size, df, transform=transform, split='train')
    
    def training_step(self, batch, batch_idx):
        (x1, x2), labels = batch
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)
        z1 = self.model(x1)
        z2 = self.model(x2)
        loss = self.loss(z1, z2)
        return loss

    def configure_optimizers(self):
        max_epochs = int(self.config.epochs)
        param_groups = define_param_groups(self.model, self.config.weight_decay, 'adam')
        lr = self.config.lr
        optimizer = Adam(param_groups, lr=lr, weight_decay=self.config.weight_decay)

        print(f'Optimizer Adam, '
              f'Learning Rate {lr}, '
              f'Effective batch size {self.config.batch_size * self.config.gradient_accumulation_steps}')

        
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                      milestones=[int(self.config.epochs*0.6),
                                                                  int(self.config.epochs*0.8)],
                                                      gamma=0.1)
        
        return optimizer, lr_scheduler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config', type=str, default='base', choices=configs.keys(), help='Configuration to use')
    parser.add_argument('-t','--train', action="store_true",  help='True to train, False to save weights only')


    args = parser.parse_args()
    config = configs[args.config]()
    print('Loaded configuration for', args.config)
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

    print(f'Saving backbone weights loaded from {pre_trainer.best_checkpoint}..')
    best_pretrain = weights_update(model, pre_trainer.best_checkpoint)

    resnet_backbone_weights = best_pretrain.model.backbone
    torch.save(resnet_backbone_weights.state_dict(), f'{config.save_prefix}_{config.df}_backbone_weights.ckpt')
    # torch.save({
    #             'model_state_dict': resnet_backbone_weights.state_dict(),
    #             }, f'{config.save_prefix}_{config.df}_backbone_weights.ckpt')

    print('Checkpoint saved')

if __name__=="__main__":
    main()
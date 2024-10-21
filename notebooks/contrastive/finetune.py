import numpy as np
import os
import torch
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
from pretrain import Trainer
import wandb

class Hparams:
    def __init__(self):
        self.epochs = 100 # number of training epochs
        self.seed = 42 # randomness seed
        self.cuda = True # use nvidia gpu
        self.img_size = 224 #image shape
        self.save = "./saved_models/" # save checkpoint
        self.gradient_accumulation_steps = 1 # gradient accumulation steps
        self.batch_size = 500
        self.lr = 0.1#1e-3
        self.embedding_size= 4*128 # papers value is 128
        self.temperature = 0.5 # 0.1 or 0.5
        self.df='tinyimagenet' #imagenet1k_0.1
        self.random = True
        self.backbone ='resnet50'
        self.pretrained_exp = 'SimCLR_pretrain_resnet50tinyimagenet'
        self.ckpt = 'resnet50_tinyimagenet_backbone_weights.ckpt'
        self.dataset_path = "/l/users/hawau.toyin/CV805/OptimizingPretext/datasets/zh-plus___tiny-imagenet"
        self.resume_from_checkpoint = False
        self.reduce = 1.0
        self.linear_eval = True
     
class SimCLR_eval(nn.Module):
    def __init__(self, lr, model=None, linear_eval=False, feat_dim=2048):
        super().__init__()
        self.lr = lr
        self.linear_eval = linear_eval
        if self.linear_eval:
          model.eval()
          model.requires_grad_(False)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(feat_dim,1000),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(0.1),
            # torch.nn.Linear(128, 10)
        )

        self.model = torch.nn.Sequential(
            model, self.mlp
        )
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, X):
        return self.model(X)   
        
class FinetuneTrainer(Trainer):
    def __init__(self, config, model):
        # super().__init__(config, model)
        self.config = config
        self.config.batch_size = self.config.batch_size * torch.cuda.device_count()
        self.model = model 
        # self.exp_name = "SimCLR_finetune_" + self.config.pretrained_exp + self.config.df 
        self.exp_name = "SimCLR_linearprobe_resnet50_random" +  self.config.df 
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = self.configure_optimizers()
        self.best_loss = np.inf
        self.best_acc = 0
        self.epoch = 0
        self.device = "cuda"
        self.best_checkpoint = f"{self.config.save}/{self.exp_name}/best.ckpt"
        self.global_step = 0
        
        os.makedirs(f"{self.config.save}/{self.exp_name}", exist_ok=True)
        
        self.model = self.model.to(self.device)
        
        if self.config.resume_from_checkpoint:
            self.resume_checkpoint()
        # if torch.cuda.device_count() > 1:
        #     print("Using", torch.cuda.device_count(), "GPUs")
        #     self.model = torch.nn.DataParallel(self.model)
        
        print(f'Number of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6:.2f}M')
        print(f'Number of total parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M')
        
        self.train_loader = self.get_dataloaders()
        self.test_loader = self.get_dataloaders(split='validation')
        
        self.logger = WandBLogger(project='cOptimizingPretext', name=self.exp_name, config=config.__dict__)
    
    def train(self):
        for epoch in range(self.epoch, self.config.epochs):
            torch.cuda.empty_cache()
            epoch_loss = 0
            epoch_acc = 0
            self.model.train()
            tq_obj = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config.epochs}', total=len(self.train_loader), leave=False)
            for i, batch in enumerate(tq_obj):
                self.optimizer.zero_grad()
                loss, acc = self.training_step(batch, i)
                tq_obj.set_postfix({'loss': loss.item()})
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                epoch_acc += acc
                if self.global_step % 100 == 0:
                    self.logger.log_dict({'Train loss': loss.item(), 
                                          'epoch': epoch,
                                          'Train acc': acc}, self.global_step)
                self.global_step+=1
            self.logger.log('Contrastive loss epoch', epoch_loss/len(self.train_loader),
                             self.global_step )
            if epoch_acc/len(self.train_loader) < self.best_acc:
                print(f"Saved best ckpt at epoch {epoch}, with acc {epoch_acc/len(self.train_loader)}")
                self.save_best()
                self.best_acc = epoch_acc/len(self.train_loader)
                self.logger.log('Best acc', self.best_acc, self.global_step)
            self.epoch += 1
            self.save()
            
            val_loss, val_acc = 0, 0
            val_tq = tqdm(self.test_loader, desc=f'Validation', total=len(self.test_loader), leave=True)
            for i, batch in enumerate(val_tq):
                loss, acc = self.validation_step(batch, i)
                val_loss += loss.item()
                val_acc += acc
            val_loss /= len(self.test_loader)
            val_acc /= len(self.test_loader)
            self.logger.log_dict({'Val loss': val_loss,
                                  'Val acc': val_acc}, self.global_step)
    
    def save(self, path=None):
        to_save = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_acc': self.best_acc
        }
        torch.save(to_save, path or f"{self.config.save}/{self.exp_name}/last.ckpt")
    
    def resume_checkpoint(self):
        checkpoint = torch.load(f"{self.config.save}/{self.exp_name}/last.ckpt")
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print("Loading existing checkpoint from", f"{self.config.save}/{self.exp_name}/last.ckpt")
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_acc = checkpoint.get('best_acc', 0)
    
    def get_dataloaders(self, split='train'):
        transform = Augment(self.config.img_size).test_transform
        df = load_dataset(self.config.dataset_path)
        if self.config.reduce < 1.0:
            df = reduce_dataset(df, self.config.reduce)
        return get_imagenet_dataloader(self.config.batch_size, df, transform=transform, split=split)
    
    def training_step(self, batch, batch_idx):
        x, labels = batch
        x = x.to(self.device)
        labels = labels.to(self.device)
        z = self.model(x)
        loss = self.loss(z, labels)
        
        pred = z.argmax(1)
        acc = (pred == labels).sum().item() / labels.size(0)
        return loss, acc
    
    def validation_step(self, batch, batch_idx):
        x, labels = batch
        x = x.to(self.device)
        labels = labels.to(self.device)
        z = self.model(x)
        loss = self.loss(z, labels)
        predicted = z.argmax(1)
        acc = (predicted == labels).sum().item() / labels.size(0)
        return loss, acc
    
    def validate(self):
        checkpoint = torch.load(f"{self.config.save}/{self.exp_name}/best.ckpt")
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        val_loss, val_acc = 0, 0
        val_tq = tqdm(self.test_loader, desc=f'Validation', total=len(self.test_loader), leave=True)
        for i, batch in enumerate(val_tq):
            loss, acc = self.validation_step(batch, i)
            val_loss += loss.item()
            val_acc += acc
        val_loss /= len(self.test_loader)
        val_acc /= len(self.test_loader)
        print(f"Validation loss: {val_loss}, Validation accuracy: {val_acc}")
    
    def configure_optimizers(self):
        if self.config.linear_eval:
          print(f"\n\n Attention! Linear evaluation \n")
          optimizer = SGD(self.model.mlp.parameters(), lr=self.config.lr, momentum=0.9)
        else:
          optimizer = SGD(self.model.model.parameters(), lr=self.config.lr, momentum=0.9)
        return optimizer
    

config = Hparams()
reproducibility(config)
backbone = models.resnet50(pretrained=False)
backbone.fc = nn.Identity()
feat_dim = 2048
if not config.random:
    print('Loading the pretrained model')
    checkpoint = torch.load(config.ckpt)
    backbone.load_state_dict(checkpoint['model_state_dict'])
    
model = SimCLR_eval(config.lr, model=backbone, linear_eval=True)

trainer = FinetuneTrainer(config, model)

trainer.train()

trainer.validate()



import torch
import torchvision.models as models
import numpy as np
import os
import torch
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import STL10
import torchvision.transforms as T

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim import SGD, Adam
from utils import *

import torch
from pytorch_lightning import Trainer
import os
from pytorch_lightning.callbacks import GradientAccumulationScheduler
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.models import  resnet18, resnet50
import warnings
warnings.filterwarnings("ignore")
from pytorch_lightning.loggers import WandbLogger


class AddProjection(nn.Module):
    def __init__(self, config, model=None, mlp_dim=512):
        super(AddProjection, self).__init__()
        embedding_size = config.embedding_size
        self.backbone = default(model, models.resnet18(pretrained=False, num_classes=config.embedding_size))
        mlp_dim = default(mlp_dim, self.backbone.fc.in_features)
        print('Dim MLP input:',mlp_dim)
        self.backbone.fc = nn.Identity()

        # add mlp projection head
        self.projection = nn.Sequential(
            nn.Linear(in_features=mlp_dim, out_features=mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(),
            nn.Linear(in_features=mlp_dim, out_features=embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

    def forward(self, x, return_embedding=False):
        embedding = self.backbone(x)
        if return_embedding:
            return embedding
        return self.projection(embedding)



def define_param_groups(model, weight_decay, optimizer_name):
    def exclude_from_wd_and_adaptation(name):
        if 'bn' in name:
            return True
        if optimizer_name == 'lars' and 'bias' in name:
            return True

    param_groups = [
        {
            'params': [p for name, p in model.named_parameters() if not exclude_from_wd_and_adaptation(name)],
            'weight_decay': weight_decay,
            'layer_adaptation': True,
        },
        {
            'params': [p for name, p in model.named_parameters() if exclude_from_wd_and_adaptation(name)],
            'weight_decay': 0.,
            'layer_adaptation': False,
        },
    ]
    return param_groups


class SimCLR_pl(pl.LightningModule):
    def __init__(self, config, model=None, feat_dim=512):
        super().__init__()
        self.config = config

        self.model = AddProjection(config, model=model, mlp_dim=feat_dim)

        self.loss = ContrastiveLoss(config.batch_size, temperature=self.config.temperature)

    def forward(self, X):
        return self.model(X)

    def training_step(self, batch, batch_idx):
        (x1, x2), labels = batch
        z1 = self.model(x1)
        z2 = self.model(x2)
        loss = self.loss(z1, z2)
        self.log('Contrastive loss', loss,rank_zero_only=True, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        max_epochs = int(self.config.epochs)
        param_groups = define_param_groups(self.model, self.config.weight_decay, 'adam')
        lr = self.config.lr
        optimizer = Adam(param_groups, lr=lr, weight_decay=self.config.weight_decay)

        print(f'Optimizer Adam, '
              f'Learning Rate {lr}, '
              f'Effective batch size {self.config.batch_size * self.config.gradient_accumulation_steps}')

        scheduler_warmup = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=max_epochs,
                                                         warmup_start_lr=0.0)

        return [optimizer], [scheduler_warmup]
    



available_gpus = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])
save_model_path = os.path.join(os.getcwd(), "saved_models/")
print('available_gpus:',available_gpus)
filename='SimCLR_ResNet50_imagenet_'
resume_from_checkpoint = True
train_config = Hparams()

 
save_name = filename + '.ckpt'

backbone = resnet50(pretrained=False)
model = SimCLR_pl(train_config, model=backbone, feat_dim=2048)


transform = Augment(train_config.img_size)
df = load_dataset("/fsx/hyperpod-input-datasets/AROA6GBMFKRI2VWQAUGYI:Hawau.Toyin@mbzuai.ac.ae/hf_datasets/ILSVRC___imagenet-1k")
df = reduce_dataset(df, 0.3)
data_loader = get_imagenet_dataloader(train_config.batch_size, df, transform)

accumulator = GradientAccumulationScheduler(scheduling={0: train_config.gradient_accumulation_steps})
checkpoint_callback = ModelCheckpoint(filename=filename, dirpath=save_model_path,
                                        save_last=True, save_top_k=2,monitor='Contrastive loss_epoch',mode='min')

wandb_logger = WandbLogger(project="contrastive-res50", name=f"{filename}_pretrain", config=train_config.__dict__)

# wandb_logger = None
if resume_from_checkpoint:
    print('Resuming from checkpoint')
    trainer = Trainer(callbacks=[accumulator, checkpoint_callback],
                    gpus=available_gpus,
                    max_epochs=train_config.epochs,
                    resume_from_checkpoint=train_config.checkpoint_path,
                    logger=wandb_logger)
    # , strategy='ddp'
    # logger=wandb_logger,
else:
    print('Starting from scratch')
    trainer = Trainer(callbacks=[accumulator, checkpoint_callback],
                    gpus=available_gpus,
                    max_epochs=train_config.epochs,
                    logger=wandb_logger)


trainer.fit(model, data_loader)

trainer.save_checkpoint(save_name)

backbone = resnet50(pretrained=False)
model_pl = SimCLR_pl(train_config, model=backbone, feat_dim=2048)

model_pl = weights_update(model_pl, checkpoint_callback.best_model_path)

resnet_backbone_weights = model_pl.model.backbone
torch.save({
            'model_state_dict': resnet_backbone_weights.state_dict(),
            }, 'resnet50_imagenet_backbone_weights.ckpt')



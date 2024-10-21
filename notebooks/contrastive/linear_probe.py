import pytorch_lightning as pl
import torch
from torch.optim import SGD
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import GradientAccumulationScheduler
import os
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.models import resnet18, resnet50
import torchvision.models as models
from torchvision.datasets import STL10
import warnings
warnings.filterwarnings("ignore")
from pytorch_lightning.loggers import WandbLogger

from utils import *

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
        self.data='tinyimagenet' #imagenet1k_0.1
        self.random = False
        self.backbone ='resnet50'
        self.pretrained_exp = 'SimCLR_pretrain_resnet50tinyimagenet'
        self.ckpt = 'resnet50_tinyimagenet_backbone_weights.ckpt'
        
class SimCLR_eval(pl.LightningModule):
    def __init__(self, lr, model=None, linear_eval=False, feat_dim=2048):
        super().__init__()
        self.lr = lr
        self.linear_eval = linear_eval
        if self.linear_eval:
          model.eval()
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
        with torch.no_grad():
            X = self.model[0](X)
        return self.model[1](X)

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.forward(x)
        loss = self.loss(z, y)
        self.log('Cross Entropy loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        predicted = z.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        self.log('Train Acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.forward(x)
        loss = self.loss(z, y)
        self.log('Val CE loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        predicted = z.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        self.log('Val Accuracy', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        if self.linear_eval:
          print(f"\n\n Attention! Linear evaluation \n")
          optimizer = SGD(self.mlp.parameters(), lr=self.lr, momentum=0.9)
        else:
          optimizer = SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        return [optimizer]


# general stuff
available_gpus = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])
train_config = Hparams()
exp_name = f"SimCLR_{train_config.backbone}_finetune_from_resnet50tinyimagenet_on_{train_config.data}"
save_model_path = os.path.join(os.getcwd(), f"saved_models/{exp_name}")
os.makedirs(save_model_path, exist_ok=True)
print('available_gpus:', available_gpus)
reproducibility(train_config)
save_name = 'last.ckpt'

wandb_logger = WandbLogger(project="cOptimizingPretext", name=exp_name, config=train_config.__dict__)

# load resnet backbone
backbone = models.resnet50(pretrained=False)
backbone.fc = nn.Identity()
if not train_config.random:
    print('Loading the pretrained model')
    checkpoint = torch.load(train_config.ckpt)
    backbone.load_state_dict(checkpoint['model_state_dict'])
model = SimCLR_eval(train_config.lr, model=backbone, linear_eval=True)

# preprocessing and data loaders
transform_preprocess = Augment(train_config.img_size).test_transform

print('Loading the dataset')
# df = load_dataset("/l/users/emilio.villa/huggingface/datasets/ILSVRC___imagenet-1k")
# df = reduce_dataset(df, 0.1)

df = load_dataset("zh-plus/tiny-imagenet", cache_dir="datasets/")
data_loader = get_imagenet_dataloader(train_config.batch_size, df, transform_preprocess)
data_loader_test = get_imagenet_dataloader(train_config.batch_size, df, transform_preprocess, split='valid')


# callbacks and trainer
accumulator = GradientAccumulationScheduler(scheduling={0: train_config.gradient_accumulation_steps})

checkpoint_callback = ModelCheckpoint(filename='best', dirpath=save_model_path,save_last=True,save_top_k=2,
                                       monitor='Val Accuracy_epoch', mode='max')

trainer = Trainer(callbacks=[checkpoint_callback,accumulator],
                  gpus=available_gpus,
                  max_epochs=train_config.epochs, 
                  logger=wandb_logger,)

trainer.fit(model, data_loader,data_loader_test)
trainer.save_checkpoint(save_name)
trainer.validate(ckpt_path=checkpoint_callback.best_model_path, dataloaders=data_loader_test)

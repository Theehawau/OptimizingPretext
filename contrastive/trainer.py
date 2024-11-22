from tqdm import tqdm
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchtune.training.metric_logging import WandBLogger


from utils import *
from config import configs


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
    

class FinetuneTrainer(Trainer):
    def __init__(self, config, model):
        self.config = config
        self.config.batch_size = self.config.batch_size * torch.cuda.device_count()
        self.model = model 
        self.exp_name = f"_{self.config.exp_id}_resnet50_12hrs_{self.config.df}"
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer, self.scheduler = self.configure_optimizers()
        self.best_loss = np.inf
        self.best_acc = 0
        self.epoch = 0
        self.device = "cuda"
        self.best_checkpoint = f"{self.config.save}/{self.exp_name}/best.ckpt"
        self.global_step = 0
        self.es_count = 0
        
        os.makedirs(f"{self.config.save}/{self.exp_name}", exist_ok=True)
        
        self.model = self.model.to(self.device)
        
        # print(self.model)
        
        if self.config.resume_from_checkpoint:
            self.resume_checkpoint()
        
        print(f'Number of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6:.2f}M')
        print(f'Number of total parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M')
        
        self.train_loader = self.get_dataloaders()
        self.test_loader = self.get_dataloaders(split=config.test_split)
        
        self.logger = WandBLogger(project='full_FT_OptimizingPretext', name=self.exp_name, config=config.__dict__)
    
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
            self.logger.log('Loss epoch', epoch_loss/len(self.train_loader),
                             self.global_step )
            
            self.epoch += 1
            self.save()
            
            # Validation Step
            val_loss, val_acc = 0, 0
            val_tq = tqdm(self.test_loader, desc=f'Validation', total=len(self.test_loader), leave=True)
            for i, batch in enumerate(val_tq):
                loss, acc = self.validation_step(batch, i)
                val_loss += loss.item()
                val_acc += acc
            val_loss /= len(self.test_loader)
            self.scheduler.step(val_loss)
            val_acc /= len(self.test_loader)
            
            # Best checkpoint Logic
            if val_acc > self.best_acc:
                print(f"Saved best ckpt at epoch {epoch}, with acc {val_acc}")
                self.save_best()
                self.best_acc = val_acc
                self.es_count = 0                
            else:
                self.es_count += 1
                if self.es_count > self.config.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
                
            self.logger.log_dict({'Val loss': val_loss,
                                  'Val acc': val_acc,
                                  'lr': self.scheduler.get_last_lr()[0],
                                  'Best Val acc': self.best_acc}, self.global_step)
    
    def save(self, path=None):
        to_save = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
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
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        print("Loading existing checkpoint from", f"{self.config.save}/{self.exp_name}/last.ckpt")
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_acc = checkpoint.get('best_acc', 0)
        self.config.lr = self.scheduler.get_last_lr()[0]
    
    def get_dataloaders(self, split='train'):
        transform = Augment(self.config.img_size).test_transform
        df = load_dataset(self.config.dataset_path)
        if 'cls' in df['train'].features:
            df = transform_dataset_clip_benchmark(df)
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
        with torch.no_grad():
            z = self.model(x)
        labels = labels.to(self.device)
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
          print(f"\n\nAttention! Linear evaluation \n")
          optimizer = SGD(self.model.mlp.parameters(), lr=self.config.lr, momentum=0.9)
        else:
          print(f"\n\nAttention! Full Fine Tuning \n")
          optimizer = SGD(self.model.model.parameters(), lr=self.config.lr, momentum=0.9)
        scheduler = ReduceLROnPlateau(optimizer, 'min')
        return optimizer,scheduler
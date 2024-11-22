class Hparams:
    def __init__(self):
        self.epochs = 100 # number of training epochs
        self.seed = 42 # randomness seed
        self.cuda = True # use nvidia gpu
        self.device = "cuda" # use nvidia gpu
        self.img_size = 224 #image shape
        self.save = "./saved_models/" # save checkpoint
        self.gradient_accumulation_steps = 1 # gradient accumulation steps
        self.batch_size = 200
        self.lr = 0.1 #1e-3
        self.embedding_size= 4*128 # papers value is 128
        self.temperature = 0.5 # 0.1 or 0.5
        self.df='imagenet_0.3' #imagenet1k_0.1
        self.random = False
        self.backbone ='resnet50'
        self.exp_prefix='SimCLR_pretrain36hrs_'
        self.pretrained_exp = 'SimCLR_pretrain_resnet50tinyimagenet'
        self.ckpt = 'resnet50_imagenet_0.3_backbone_weights.ckpt'
        self.dataset_path = "ILSVRC/imagenet-1k"
        self.test_split = 'validation'
        self.resume_from_checkpoint = False
        self.reduce = 0.3
        self.linear_eval = False
        self.patience = 10 
        self.architecture='resnet'
        self.continue_task = False
        
class HparamsPretrain(Hparams):
    def __init__(self):
        super().__init__()
        self.epochs = 100 # number of training epochs
        self.batch_size = 200
        self.lr = 3e-3 # for ADAm only
        self.weight_decay = 1e-6
        self.dataset_path = "ILSVRC/imagenet-1k"
        self.resume_from_checkpoint = False
        self.save_prefix = 'simclr'
       
class HparamsPretrainViTs(Hparams):
    def __init__(self):
        super().__init__()
        self.epochs = 100 # number of training epochs
        self.batch_size = 150
        self.cuda = True
        self.device = "cuda"
        self.lr = 3e-3 # for ADAm only
        self.weight_decay = 1e-6
        self.dataset_path = "ILSVRC/imagenet-1k"
        self.resume_from_checkpoint = True
        self.save_prefix = 'simclr_vits_36hrs'
        self.architecture="vits"
        self.backbone ='vits'
        self.exp_prefix='SimCLR_pretrain36hrs_'

class HparamsTinyImagenet(Hparams):
    def __init__(self):
        super().__init__()
        self.batch_size = 800
        self.lr = 0.1#1e-3
        self.embedding_size= 4*128 # papers value is 128
        self.df='tinyimagenet' #imagenet1k_0.1
        self.random = False
        self.backbone ='resnet50'
        self.ckpt = 'resnet50_12hrs_imagenet_0.3_backbone_weights.ckpt'
        self.dataset_path = "zh-plus/tiny-imagenet"
        self.resume_from_checkpoint = False
        self.test_split = 'valid'
        self.reduce = 1.0
        self.linear_eval = True 

# Modify for full finetuning 

class HparamsFullFT(Hparams):
    def __init__(self, dataset="tinyimagenet"):
        super().__init__()
        self.save = "./full_FT_models/" # save checkpoint
        self.batch_size = 400
        self.exp_id = 6 # experiment id to identify which finetuning experiment
        self.lr = 0.01
        self.ckpt = "simclr_jigsaw_rotation_imagenet_0.3_backbone_weights.ckpt"
        self.resume_from_checkpoint = False
        self.reduce = 1.0
        self.random = False
        self.linear_eval = False
        
        # tinyimagenet
        if dataset == "tinyimagenet":
            self.df='tinyimagenet' 
            self.num_classes=200
            self.test_split = 'valid'
            self.dataset_path = "zh-plus/tiny-imagenet"
        
        # voc2007
        elif dataset == "voc2007":
            self.df='voc2007'
            self.num_classes=20
            self.test_split = 'test'
            self.dataset_path = "clip-benchmark/wds_voc2007"
        
        # Caltech
        elif dataset == 'caltech':
            self.df='caltech'
            self.num_classes=102
            self.test_split = 'test'
            self.dataset_path = "clip-benchmark/wds_vtab-caltech101"
            
        else:
            raise ValueError(f"Dataset {dataset} not supported")


        
configs = {
    'base': Hparams,
    'pretrain': HparamsPretrain,
    'pretrain_tinyimagenet': HparamsTinyImagenet,
    'pretrain_vits': HparamsPretrainViTs,
    'full_ft': HparamsFullFT,
}
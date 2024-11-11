class Hparams:
    def __init__(self):
        self.epochs = 100 # number of training epochs
        self.seed = 42 # randomness seed
        self.cuda = True # use nvidia gpu
        self.device = "cuda" # use nvidia gpu
        self.img_size = 224 #image shape
        self.save = "./saved_models/" # save checkpoint
        self.gradient_accumulation_steps = 1 # gradient accumulation steps
        self.batch_size = 800
        self.lr = 0.1#1e-3
        self.embedding_size= 4*128 # papers value is 128
        self.temperature = 0.5 # 0.1 or 0.5
        self.df='imagenet_0.3' #imagenet1k_0.1
        self.random = False
        self.backbone ='resnet50'
        self.exp_prefix='SimCLR_pretrain36hrs_'
        self.pretrained_exp = 'SimCLR_pretrain_resnet50tinyimagenet'
        self.ckpt = 'resnet50_imagenet_0.3_backbone_weights.ckpt'
        self.dataset_path = "/fsx/hyperpod-input-datasets/AROA6GBMFKRI2VWQAUGYI:Hawau.Toyin@mbzuai.ac.ae/hf_datasets/ILSVRC___imagenet-1k"
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
        self.cuda = False
        self.device = "cpu"
        self.lr = 3e-3 # for ADAm only
        self.weight_decay = 1e-6
        self.dataset_path = "/l/users/emilio.villa/huggingface/datasets/ILSVRC___imagenet-1k"
        self.resume_from_checkpoint = False
        self.save_prefix = 'simclr'
        # self.exp_prefix='simclr_jigsaw_'
        # self.continue_task = True
        # self.previous_task_backbone ="/l/users/hawau.toyin/CV805/OptimizingPretext/notebooks/contrastive/saved_models/pretrained_models/jigsaw_rn50_adam_best_12hrs.pth"
        
class HparamsPretrainViTs(Hparams):
    def __init__(self):
        super().__init__()
        self.epochs = 100 # number of training epochs
        self.batch_size = 150
        self.cuda = True
        self.device = "cuda"
        self.lr = 3e-3 # for ADAm only
        self.weight_decay = 1e-6
        self.dataset_path = "/l/users/emilio.villa/huggingface/datasets/ILSVRC___imagenet-1k"
        self.resume_from_checkpoint = True
        self.save_prefix = 'simclr_vits_12hrs'
        self.architecture="vits"
        self.backbone ='vits'
        self.exp_prefix='SimCLR_pretrain36hrs_'

class HparamsPretrainFromJigsaw(HparamsPretrain):
    def __init__(self):
        super().__init__()
        self.cuda = True
        self.device = "cuda"
        self.lr = 3e-3 # for ADAm only
        self.weight_decay = 1e-6
        self.resume_from_checkpoint = False
        self.save_prefix = 'simclr_jigsaw_'
        self.exp_prefix='simclr_jigsaw_'
        self.continue_task = True
        self.previous_task_backbone ="/l/users/hawau.toyin/CV805/OptimizingPretext/notebooks/contrastive/saved_models/pretrained_models/jigsaw_rn50_adam_best_12hrs.pth"


class HparamsPretrainFromRotation(HparamsPretrain):
    def __init__(self):
        super().__init__()
        self.cuda = True
        self.device = "cuda"
        self.lr = 3e-3 # for ADAm only
        self.weight_decay = 1e-6
        self.resume_from_checkpoint = False
        self.save_prefix = 'simclr_rotation_'
        self.exp_prefix='simclr_rotation_'
        self.continue_task = True
        self.previous_task_backbone ="/l/users/hawau.toyin/CV805/OptimizingPretext/notebooks/contrastive/saved_models/pretrained_models/img1k03-resnet50-best-epoch-12.pth"


class HparamsTinyImagenet(Hparams):
    def __init__(self):
        super().__init__()
        self.batch_size = 800
        self.lr = 0.1#1e-3
        self.embedding_size= 4*128 # papers value is 128
        self.df='tinyimagenet' #imagenet1k_0.1
        self.random = False
        self.backbone ='resnet50'
        #self.ckpt = 'resnet50_tinyimagenet_backbone_weights.ckpt'
        self.ckpt = 'resnet50_12hrs_imagenet_0.3_backbone_weights.ckpt'
        self.dataset_path = "zh-plus/tiny-imagenet"
        self.resume_from_checkpoint = False
        self.test_split = 'valid'
        self.reduce = 1.0
        self.linear_eval = True 
  
class HparamsImagenet1k_0_1(Hparams):
    def __init__(self):
        super().__init__()
        self.df='imagenet_0.1' #imagenet1k_0.1
        self.batch_size = 800
        self.ckpt = 'resnet50_12hrs_imagenet_0.3_backbone_weights.ckpt'
        self.resume_from_checkpoint = False
        self.reduce = 0.1
        self.linear_eval = True

class HparamsImagenet1k_0_3(Hparams):
    def __init__(self):
        super().__init__()
        self.df='imagenet_0.3' #imagenet1k_0.1
        self.batch_size = 800
        self.ckpt = 'resnet50_imagenet_0.3_backbone_weights.ckpt'
        self.resume_from_checkpoint = False
        self.reduce = 0.3
        self.linear_eval = False

        
configs ={
    'base': Hparams,
    'tinyimagenet': HparamsTinyImagenet,
    'imagenet1k_0.1': HparamsImagenet1k_0_1,
    'imagenet1k_0.3': HparamsImagenet1k_0_3,
    'pretrain': HparamsPretrain,
    'pretrain_from_jigsaw': HparamsPretrainFromJigsaw,
    'pretrain_from_rotation': HparamsPretrainFromRotation,
    'pretrain_vits': HparamsPretrainViTs
}
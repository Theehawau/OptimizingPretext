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
        self.save_prefix = 'simclr_vits_36hrs'
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
        self.save_prefix = 'simclr_24hr'
        self.exp_prefix='simclr_24hr'
        self.continue_task = True
        # self.previous_task_backbone ="/l/users/hawau.toyin/CV805/OptimizingPretext/notebooks/contrastive/jigsaw_rotation_resnet50.pth"
        self.previous_task_backbone="/l/users/hawau.toyin/CV805/OptimizingPretext/notebooks/contrastive/simclr_12hrs_backbone.pt"


class HparamsPretrainFromRotation(HparamsPretrain):
    def __init__(self):
        super().__init__()
        self.cuda = True
        self.device = "cuda"
        self.lr = 3e-3 # for ADAm only
        self.weight_decay = 1e-6
        self.resume_from_checkpoint = False
        self.save_prefix = 'simclr_rotation_jigsaw_'
        self.exp_prefix='simclr_rotation_jigsaw_'
        self.continue_task = True
        self.previous_task_backbone ="/l/users/hawau.toyin/CV805/OptimizingPretext/notebooks/contrastive/s1-jigsaw-s2-rot-bs64-img1k03-resnet50-best-epoch-15.pth"

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


class HparamsFullFT(Hparams):
    def __init__(self, dataset="tinyimagenet"):
        super().__init__()
        self.save = "./full_FT_models/" # save checkpoint
        self.batch_size = 400
        self.exp_id = 6
        self.lr = 0.01
        self.ckpt = "/l/users/hawau.toyin/CV805/OptimizingPretext/notebooks/contrastive/simclr_jigsaw_rotation_imagenet_0.3_backbone_weights.ckpt"
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

class HparamsFullFT1(HparamsFullFT):
    def __init__(self):
        super().__init__()
        self.exp_id = 1
        self.random = True

class HparamsFullFT8(HparamsFullFT):
    def __init__(self):
        super().__init__()
        self.exp_id=8
        self.ckpt="/l/users/hawau.toyin/CV805/OptimizingPretext/notebooks/contrastive/s1-simclr-s2-jigsaw-s3-rot-bs64-img1k03-resnet50-best-epoch-10.pth"

class HparamsFullFT9(HparamsFullFT):
    def __init__(self):
        super().__init__()
        self.exp_id=9
        self.ckpt='/l/users/hawau.toyin/CV805/OptimizingPretext/notebooks/contrastive/s1-jigsaw-s2-simclr-s3-rot-bs64-img1k03-resnet50-best-epoch-13.pth'
class HparamsFullFT10(HparamsFullFT):
    def __init__(self):
        super().__init__()
        self.exp_id=10
        self.ckpt="/l/users/hawau.toyin/CV805/OptimizingPretext/notebooks/contrastive/simclr_rotation_jigsaw__imagenet_0.3_backbone_weights.ckpt"

class HparamsFullFT2(HparamsFullFT):
    def __init__(self):
        super().__init__()
        self.exp_id=2 
        self.ckpt="/l/users/hawau.toyin/CV805/OptimizingPretext/notebooks/contrastive/36hour3-rotation-img1k03-resnet50-best-epoch-30.pth"
        
class HparamsFullFT3(HparamsFullFT):
    def __init__(self):
        super().__init__()
        self.exp_id=3  
        self.ckpt="/l/users/hawau.toyin/CV805/OptimizingPretext/notebooks/contrastive/SimCLRresnet50_36hrs_imagenet_0.3_backbone_weights.ckpt"
        
class HparamsFullFT4(HparamsFullFT):
    def __init__(self):
        super().__init__()
        self.exp_id=4          
        self.ckpt="/l/users/hawau.toyin/CV805/OptimizingPretext/notebooks/contrastive/jigsaw_rn50_36ep_best.pth"

class HparamsFullFT7(HparamsFullFT):
    def __init__(self):
        super().__init__()
        self.exp_id=7        
        self.ckpt="/l/users/hawau.toyin/CV805/OptimizingPretext/notebooks/contrastive/s1-simclr-s2-rot-s3-jigsaw_rn50.pth"

class HparamsFullFT5(HparamsFullFT):
    def __init__(self):
        super().__init__()
        self.exp_id=5            
        self.ckpt="/l/users/hawau.toyin/CV805/OptimizingPretext/notebooks/contrastive/s1-rot-s2-simclr-s3-jig_rn50.pth"
        
configs = {
    'base': Hparams,
    'tinyimagenet': HparamsTinyImagenet,
    'imagenet1k_0.1': HparamsImagenet1k_0_1,
    'imagenet1k_0.3': HparamsImagenet1k_0_3,
    'pretrain': HparamsPretrain,
    'pretrain_from_jigsaw': HparamsPretrainFromJigsaw,
    'pretrain_from_rotation': HparamsPretrainFromRotation,
    'pretrain_vits': HparamsPretrainViTs,
    'full_ft': HparamsFullFT,
#     '1': HparamsFullFT1,
#     '2': HparamsFullFT2,
#     '8':HparamsFullFT8,
#     '9':HparamsFullFT9,
#     '10':HparamsFullFT10,
#     '3':HparamsFullFT3,
#     '4':HparamsFullFT4,
#     '7':HparamsFullFT7,
#     '5':HparamsFullFT5
}
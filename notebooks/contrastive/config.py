class Hparams:
    def __init__(self):
        self.epochs = 100 # number of training epochs
        self.seed = 42 # randomness seed
        self.cuda = True # use nvidia gpu
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
        self.pretrained_exp = 'SimCLR_pretrain_resnet50tinyimagenet'
        self.ckpt = 'resnet50_imagenet_0.3_backbone_weights.ckpt'
        self.dataset_path = "/fsx/hyperpod-input-datasets/AROA6GBMFKRI2VWQAUGYI:Hawau.Toyin@mbzuai.ac.ae/hf_datasets/ILSVRC___imagenet-1k"
        self.test_split = 'validation'
        self.resume_from_checkpoint = False
        self.reduce = 0.3
        self.linear_eval = False
        self.patience = 10 
        
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
}
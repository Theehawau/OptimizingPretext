from yacs.config import CfgNode as CN

_C = CN()

_C.name = 'optimize-pretext'
_C.seed = 42
_C.cache_dir = '/tmp/'
_C.backbone = 'resnet50'
_C.resume_from_checkpoint = False



# logging and saving
_C.save = "./saved_models/"
# _C.output.log_dir = 'logs'
# _C.output.checkpoint_dir = 'checkpoints' 


# dataset configs
_C.data = CN()

_C.df = 'tinyimagenet'
_C.dataset_path = 'datasets/zh-plus/tiny-imagenet'
_C.img_size = 224


# data loader configs
_C.pretext = CN()
_C.pretext.name = ['rotation', 'jigsaw', 'contrastive']
_C.pretext.task = 'all'


# model configs
_C.model = CN()
_C.model.name = 'resnet50'
_C.model.cache_dir = 'outputs'
_C.model.input_size = 40
_C.model.hidden_size = 64
_C.model.back_end = ['rotation', 'jigsaw', 'contrastive']

# training configs
_C = CN()
_C.losses = []
_C.device = 'cuda'
_C.epochs = 100


# criterion configs
_C.criterion = CN()
_C.criterion.name = ['rotation', 'jigsaw', 'contrastive', 'classification']


#  dataloader
_C.batch_size = 200
_C.num_workers = 4


# scheduler
_C.lr = 3e-3
_C.weight_decay = 1e-6
_C.embedding_size = 4*128
_C.temperature = 0.5
_C.scheduler = 'plateau'
_C.factor = 0.1
_C.patience = 3
_C.lr_step = 10
_C.gamma = 0.5
_C.lr_decay = 0.1
_C.T_max = 10
_C.eta_min = 0.0001
_C.min_lr = 0.001
_C.milestones = [50, 100, 150]



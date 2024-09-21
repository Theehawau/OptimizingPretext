from yacs.config import CfgNode as CN

_C = CN()

_C.name = 'optimize-pretext'
_C.seed = 42
_C.cache_dir = '/tmp/'


# logging and saving
_C.output = CN()
_C.output.save_dir = 'outputs/'
_C.output.log_dir = 'logs'
_C.output.checkpoint_dir = 'checkpoints' 


# dataset configs
_C.data = CN()

_C.data.name = 'tiny-imagenet'
_C.data.root = 'zh-plus/tiny-imagenet'
_C.data.cache_dir = 'datasets/'
_C.data.size = (64,64)
_C.data.patch_ratio = 3


# data loader configs
_C.pretext = CN()
_C.pretext.name = ['rotation', 'jigsaw', 'colorization']
_C.pretext.task = 'all'


# model configs
_C.model = CN()
_C.model.name = 'resnet-50'
_C.model.cache_dir = 'outputs'
_C.model.input_size = 40
_C.model.hidden_size = 64
_C.model.back_end = ['rotation', 'jigsaw', 'colorization']

# training configs
_C.solver = CN()
_C.solver.losses = []
_C.solver.device = 'cuda'
_C.solver.epochs = 100


# criterion configs
_C.criterion = CN()
_C.criterion.name = ['rotation', 'jigsaw', 'colorization', 'classification']


#  dataloader
_C.solver.batch_size = 32
_C.solver.num_workers = 4


# scheduler
_C.solver.scheduler = 'plateau'
_C.solver.factor = 0.1
_C.solver.patience = 3
_C.solver.lr_step = 10
_C.solver.gamma = 0.5
_C.solver.lr_decay = 0.1
_C.solver.T_max = 10
_C.solver.eta_min = 0.0001
_C.solver.min_lr = 0.001
_C.solver.milestones = [50, 100, 150]



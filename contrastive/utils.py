import os
import random
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.models as models
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.multiprocessing import cpu_count
from datasets import load_dataset,DatasetDict
import matplotlib.pyplot as plt
from tqdm import tqdm

datasets_dict = {
    'imagenet_0.3': {
        'path': '/fsx/hyperpod-input-datasets/AROA6GBMFKRI2VWQAUGYI:Hawau.Toyin@mbzuai.ac.ae/hf_datasets/ILSVRC___imagenet-1k',
        'split': 'validation',
    },
    'tinyimagenet': {
        'path': '/l/users/emilio.villa/huggingface/datasets/ILSVRC___imagenet-1k',
        'split': 'valid',
    }
}

class SimCLR_pl(nn.Module):
    def __init__(self, config, model=None, feat_dim=512):
        super().__init__()
        self.config = config

        self.model = AddProjection(config, model=model, mlp_dim=feat_dim)

        # self.loss = ContrastiveLoss(config.batch_size, temperature=self.config.temperature)

    def forward(self, X):
        return self.model(X)

class AddProjection(nn.Module):
    def __init__(self, config, model=None, mlp_dim=512):
        super(AddProjection, self).__init__()
        embedding_size = config.embedding_size
        self.backbone = default(model, models.resnet50(pretrained=False, num_classes=config.embedding_size))
        if config.architecture == 'resnet':
            mlp_dim = default(mlp_dim, self.backbone.fc.in_features)
            self.backbone.fc = nn.Identity()
        
        elif config.architecture == 'vits':
            mlp_dim = 768
            self.backbone.heads = nn.Identity()

        print('Dim MLP input:',mlp_dim)

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

def imshow(img):
    """
    shows an imagenet-normalized image on the screen
    """
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    unnormalize = T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    npimg = unnormalize(img).numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def default(val, def_val):
    return def_val if val is None else val

def reproducibility(config):
    SEED = int(config.seed)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    if (config.cuda):
        torch.cuda.manual_seed(SEED)


def device_as(t1, t2):
    """
    Moves t1 to the device of t2
    """
    return t1.to(t2.device)


# From https://github.com/PyTorchLightning/pytorch-lightning/issues/924
def weights_update(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f'Checkpoint {checkpoint_path} was loaded')
    return model

def reduce_dataset(dataset_dict, proportion, seed=42):
    '''
    Reduce the dataset to a specified proportion while maintaining balanced labels.
 
    '''
    def reduce_split(dataset, proportion, seed):
 
        labels = dataset['label']
        label_indices = {}
        for idx, label in enumerate(labels):
            if label not in label_indices:
                label_indices[label] = []
            label_indices[label].append(idx)
 
        random.seed(seed)
 
        reduced_indices = []
        for label, indices in label_indices.items():
            num_samples = int(len(indices) * proportion)
            num_samples = min(num_samples, len(indices))
            reduced_indices.extend(random.sample(indices, num_samples))
 
        random.shuffle(reduced_indices)
 
        return dataset.select(reduced_indices)
 
    reduced_dict = {}
    for split_name, split_dataset in dataset_dict.items():
        reduced_dict[split_name] = reduce_split(split_dataset, proportion, seed)
 
    return DatasetDict(reduced_dict)

def transform_dataset_clip_benchmark(dataset_dict):
    '''
    Transforms the datasets from clip-benchmark to the format we are using & drops unuseful columns.
    '''
    def transform_split(split):
        # Rename columns
        split = split.rename_column("cls", "label").rename_column("webp", "image")
        # Select only 'label' and 'image' columns
        split = split.remove_columns([col for col in split.column_names if col not in {"label", "image"}])
        return split

    return DatasetDict({split_name: transform_split(split) for split_name, split in dataset_dict.items()})

class Augment:
    """
    A stochastic data augmentation module
    Transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, img_size, s=1):
        color_jitter = T.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        # 10% of the image
        blur = T.GaussianBlur((3, 3), (0.1, 2.0))

        self.train_transform = T.Compose(
            [
            T.RandomResizedCrop(size=img_size),
            T.RandomHorizontalFlip(p=0.5),  # with 0.5 probability
            T.RandomApply([color_jitter], p=0.8),
            T.RandomApply([blur], p=0.5),
            T.RandomGrayscale(p=0.2),
            # imagenet stats
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

        self.test_transform = T.Compose(
            [
                T.Resize((224,224)),
                # T.RandomResizedCrop(size=img_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)

class ImageNetHF(Dataset):
    """Custom Dataset for image loading and processing using PyTorch operations. 
    This dataset uses the Hugging Face datasets library to load the dataset.
    """
    def __init__(self, dataset, transform=None, split='train'):
        self.dataset = dataset[split]
        self.transform = transform
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Retrieve an image and apply transformations."""
        image= self.dataset[idx]['image'].convert('RGB') 
        label = self.dataset[idx]['label']
        img = self.transform(image)
        return img, label
    
    
def get_imagenet_dataloader(batch_size, dataset, transform=None, split='train'):
    imagenet = ImageNetHF(dataset, transform=transform, split=split)
    return DataLoader(imagenet, batch_size=batch_size, shuffle=True, num_workers=0)

    # , drop_last=True

def collate_fn(batch):
    pass

class ContrastiveLoss(nn.Module):
    """
    Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper
    """
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()

    def calc_similarity_batch(self, a, b):
        representations = torch.cat([a, b], dim=0)
        return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    def forward(self, proj_1, proj_2):
        """
        proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
        where corresponding indices are pairs
        z_i, z_j in the SimCLR paper
        """
        batch_size = proj_1.shape[0]
        z_i = F.normalize(proj_1, p=2, dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)

        similarity_matrix = self.calc_similarity_batch(z_i, z_j)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)

        denominator = device_as(self.mask, similarity_matrix) * torch.exp(similarity_matrix / self.temperature)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * self.batch_size)
        return loss
    

def extract_features(
    loader,
    feature_extraction_model,
    batch_size = 512,
    device = 'cuda'):
    '''
    Generates numpy array with features given a pretrained resnet model.
    Parameters:
    - loader : DataLoader object
    - feature_extraction_model : Module to extract features
    - batch_size
    Returns
    - features : Numpy array with extracted features
    '''
    # loader = DataLoader(
    #         image_dataset, batch_size=batch_size, shuffle=False, num_workers=1
    #     )
    feat_list = []
    y_list = [] #lazy way to also extract labels
    # counter = 0 ## delete
    feature_extraction_model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader): # update this if dataloader is replaced
            images = images.to(device)
            outputs = feature_extraction_model(images)
            feat_list.append(outputs)
            y_list.append(labels)
            
    return torch.cat(feat_list).cpu().numpy(), torch.cat(y_list).cpu().numpy()


def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Inputs:
        x - torch.Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5) # [B, H', W', C, p_H, p_W]
    x = x.flatten(1,2)              # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2,4)          # [B, H'*W', C*p_H*p_W]
    return x

# if __name__ == "__main__":

#     print("Loading dataset")

#     df = load_dataset("/l/users/emilio.villa/huggingface/datasets/ILSVRC___imagenet-1k")
#     df = reduce_dataset(df, 0.1)

#     dataset = ImageNetHF(df, transform=Augment(224), split='train')


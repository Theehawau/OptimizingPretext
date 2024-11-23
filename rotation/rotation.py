'''
Auxiliary script for functions/modules for training Rotation PTT
'''

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import models
from PIL import Image
import numpy as np
import random
import torch.nn.functional as F

def rotate_img(img, rot):
    if rot == 0:  #0 degrees rotation
        return img
    elif rot == 90:  #90 degrees rotation
        return np.flipud(np.transpose(img, (1, 0, 2)))
    elif rot == 180:  #180 degrees rotation
        return np.fliplr(np.flipud(img))
    elif rot == 270:  #270 degrees rotation
        return np.transpose(np.flipud(img), (1, 0, 2))
    else:
        raise ValueError('Rotation should be 0, 90, 180, or 270 degrees.')

class RotationDataset(data.Dataset):
    def __init__(self, hf_dataset, transform=None, architecture='resnet'):
        """
        Input:
            hf_dataset: HuggingFace Dataset object.
            transform: Optional transform to be applied on a sample.
        """
        self.dataset = hf_dataset
        self.transform = transform
        self.rotations = [0, 90, 180, 270]
        self.architecture = architecture

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        #load img from the HuggingFace dataset and convert to RGB
        image = self.dataset[idx]['image'].convert('RGB')  
        image = image.resize((224, 224)) 

        #create 4 rotated versions of the img (0, 1, 2, 3 for 0°, 90°, 180°, 270°)
        rotated_imgs = []
        for rot in self.rotations:
            rotated_image = rotate_img(np.array(image), rot)  #apply rotation
            rotated_image = Image.fromarray(rotated_image)    #convert back to PIL Image
            rotated_image = self.transform(rotated_image)     #apply transformations
            rotated_imgs.append(rotated_image)
        rotation_labels = torch.LongTensor([0, 1, 2, 3])

        rotated_imgs_tensor = torch.stack(rotated_imgs, dim=0)  #shape: [4, 3, H, W] for 4 rotations

        return rotated_imgs_tensor, rotation_labels

class RotationNet(nn.Module):
    def __init__(self,
                 n_rotations=4,  
                 architecture = 'resnet', 
                ):
        super(RotationNet, self).__init__()
        if architecture=='resnet':
            self.backbone = models.resnet50()
            self.backbone.fc = nn.Identity() 
            feature_dim = 2048

        elif architecture=='vit':
            self.backbone = models.vit_b_16(pretrained=False)
            self.backbone.heads = nn.Identity() 
            feature_dim = 768  

        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 128),  
            nn.ReLU(),
            nn.Linear(128, n_rotations)
        )
    def forward(self, x):
        #x shape: [batch_size, 3, 64, 64]
        features = self.backbone(x)  #shape: [batch_size, feature_dim]
        out = self.fc(features)  #shape: [batch_size, n_rotations]
        return out
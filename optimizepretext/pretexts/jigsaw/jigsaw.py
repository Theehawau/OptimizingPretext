'''
Auxiliary script for functions/modules for training Jigsaw PTT

TODO:
- Implement ViT
- Implement ResNet50 instead of 18 -> requires to update dimensions in the model
'''

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import models
import numpy as np
import random

def generate_permutations(n_permutations, n_tiles):
    """
    Generates a list of permutations, these will essentially be the 'gold truth' labels
    """
    permutations = []
    seen = set()
    while len(permutations) < n_permutations:
        perm = tuple(np.random.permutation(n_tiles))
        if perm not in seen:
            permutations.append(perm)
            seen.add(perm)
    return permutations

class JigsawPuzzleDataset(data.Dataset):
    def __init__(self, hf_dataset, permutations, transform=None):
        """
        Paramteres:
            hf_dataset: HuggingFace Dataset object.
            permutations: List of permutations.
            transform: Optional transform to be applied on a sample.
        """
        self.dataset = hf_dataset
        self.permutations = permutations
        self.n_permutations = len(permutations)
        self.n_tiles = 9  # 3x3 grid -> this can be modified later
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Load image from the HuggingFace dataset and convert to RGB
        image = self.dataset[idx]['image'].convert('RGB')  # Ensure image is in RGB

        # resize it to 255x255 we will reduce to 64x64 patches
        image = image.resize((255, 255))

        # divide the image into 3x3 grid of tiles (85x85 pixels each)
        tiles = []
        tile_size = 85 # 85 * 3 = 255

        ## Iterate over possible tiles and create the patches
        for i in range(3):
            for j in range(3):

                '''
                Original paper explanation: We randomly crop a 225 × 225 pixel window from an image (red dashed box), divide it into a 3 × 3 grid, and randomly pick a 64 × 64 pixel tiles from each 75 × 75 pixel cell.
                '''

                # Get boundaries and crop
                left = j * tile_size
                upper = i * tile_size
                right = left + tile_size
                lower = upper + tile_size
                tile = image.crop((left, upper, right, lower))
                
                # Random crop of 64x64 pixels with random shifts
                shift_max = tile_size - 64  # Max shift to introduce randomness
                left_shift = random.randint(0, shift_max)
                upper_shift = random.randint(0, shift_max)
                tile = tile.crop((left_shift, upper_shift, left_shift + 64, upper_shift + 64))
                
                # Apply any transform passed as argument
                if self.transform is not None:
                    tile = self.transform(tile)
                    
                tiles.append(tile)
                
        # Select a random permutation from the pre-computed permutations
        perm_idx = random.randint(0, self.n_permutations - 1)
        perm = self.permutations[perm_idx]

        # Shuffle the tiles according to the permutation
        shuffled_tiles = [tiles[p] for p in perm]

        # Stack tiles into a tensor
        tiles_tensor = torch.stack(shuffled_tiles, dim=0)  # Shape: [9, 3, 64, 64]

        # Return the shuffled tiles and the permutation index which is the gold label we aim the model to predict
        return tiles_tensor, perm_idx
    
class JigsawNet(nn.Module):
    def __init__(self, 
                 n_permutations,
                 architecture = 'resnet18', # 'resnet' or 'vit'
                ):
        
        super(JigsawNet, self).__init__()

        if architecture=='resnet18':
            # Backbone ResNet model 
            self.resnet = models.resnet18(weights=None) 
            self.resnet.fc = nn.Identity()  #Remove the classification layer
            resnet_output_dim = 512
        if architecture=='resnet50':
            # Backbone ResNet model 
            self.resnet = models.resnet50(weights=None) 
            self.resnet.fc = nn.Identity()  #Remove the classification layer
            resnet_output_dim = 2048
            
        elif architecture=='vit':
            pass ##TODO

        # Fully connected layers << to dispose after the PTT
        self.fc = nn.Sequential(
            nn.Linear(resnet_output_dim * 9, 4096), # each genertaes a 512-dimensional vector
            nn.ReLU(),
            nn.Linear(4096, n_permutations)
        )

    def forward(self, x):
        # x shape: [batch_size, 9, 3, 64, 64]
        batch_size = x.size(0)

        # combine batch and tile dimensions (siamese network -> feed the same weights all the patches at once)
        x = x.view(batch_size * 9, 3, 64, 64)  
        features = self.resnet(x)  # Shape: [batch_size * 9, 512]

        # concatenate the patches before the linear layers that learns to predict the permutation
        features = features.view(batch_size, 9 * 512)  # shape -> [batch_size, 9 * 512]

        #
        out = self.fc(features)  # shape: [batch_size, n_permutations]
        return out
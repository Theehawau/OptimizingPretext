'''
Script with functions to train JigSaw
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms
import numpy as np
import random
from tqdm import tqdm
from datasets import load_dataset
import datasets
from pathlib import Path
import logging
import os

from jigsaw import generate_permutations, JigsawPuzzleDataset, JigsawNet

from data_utils import reduce_dataset

def run_training_jigsaw(config):
    '''
    Runs a training loop for the JigSaw PTT
    Parameters:
    - config : dictionary with the configuration for the experiment
    '''

    ## device and seeds for reproducibility
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    ## read data
    download_path = config['dataset_download_path']
    datasets.config.DOWNLOADED_DATASETS_PATH = Path(download_path)
    access_token = config['hf_access_token']
    dataset = load_dataset(config['ptt_dataset_name'], token=access_token, cache_dir=download_path)

    if 'training_data_proportion' in config:  ## reduce training data (for imagenet1k)
        logging.info(f"Reducing training data to {config['training_data_proportion']}...")
        if config['training_data_proportion'] < 1.0:
            dataset = reduce_dataset(
                dataset,
                proportion=config['training_data_proportion'],
                seed=config['seed'])

    ## preprocess data and create dataloaders
    n_permutations = config['n_permutations']  ## essentially the number of classes
    n_tiles = 9  # for the moment we leave fixed at 9
    permutations = generate_permutations(n_permutations, n_tiles)
    permutations_dict = {i: perm for i, perm in enumerate(permutations)}

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = JigsawPuzzleDataset(dataset['train'], permutations, transform=transform)

    validation_name = 'valid' if config['ptt_dataset_name'] == 'zh-plus/tiny-imagenet' else 'validation'  # hotfix because the validation partition in imagenet is called 'valid'
    valid_dataset = JigsawPuzzleDataset(dataset[validation_name], permutations, transform=transform)

    train_loader = data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    valid_loader = data.DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

    ## instance JigsSaw model
    model = JigsawNet(
        n_permutations = n_permutations, 
        architecture = config['jigsaw_architecture'],
        backbone_weights_path = config.get('backbone_weights_path',None)
        )
    model = model.to(device)

    ## instance loss and optimizer
    criterion = nn.CrossEntropyLoss()

    if config.get('optimizer','sgd') == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['optim_lr'], momentum=config['optim_momentum'], weight_decay=config['optim_decay'])
    elif config['optimizer']=='adam':
        optimizer = optim.AdamW(model.parameters(), lr=config['optim_lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)


    num_epochs = config['epochs']
    log_each = config['log_each']

    best_accuracy = 0.0
    start_epoch = 0
    per_epoch_loss = []

    experiment_name = config['experiment_name']

    ## create a save directory if does not exist
    save_dir = config['savedir']
    os.makedirs(save_dir, exist_ok=True)
    last_checkpoint_path = os.path.join(save_dir, f'{experiment_name}_last_checkpoint.pth')
    best_checkpoint_path = os.path.join(save_dir, f'{experiment_name}_best_checkpoint.pth')

    ## load checkpoint if provided
    if 'checkpoint_dir' in config and config['checkpoint_dir'] is not None:
        checkpoint_path = config['checkpoint_dir']
        if os.path.isfile(checkpoint_path):
            logging.info(f"Loading checkpoint '{checkpoint_path}'")
            # print(f"Loading checkpoint '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
            best_accuracy = checkpoint['best_accuracy']
            per_epoch_loss = checkpoint['per_epoch_loss']
        else:
            logging.info(f"No checkpoint found at '{checkpoint_path}'")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        avg_loss = 0.0
        for batch_idx, (tiles, perm_idx) in enumerate(tqdm(train_loader)):
            tiles = tiles.to(device)  # Shape: [batch_size, 9, 3, 64, 64]
            perm_idx = perm_idx.to(device)  # Shape: [batch_size]

            optimizer.zero_grad()
            outputs = model(tiles)  # Shape: [batch_size, n_permutations]
            loss = criterion(outputs, perm_idx)
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            if batch_idx % log_each == log_each - 1:
                logging.info(f'Epoch [{epoch}/{num_epochs}], Batch [{batch_idx+1}], Loss: {avg_loss / log_each:.4f}')
                per_epoch_loss.append(avg_loss / log_each)
                avg_loss = 0.0

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for tiles, perm_idx in tqdm(valid_loader):
                tiles = tiles.to(device)
                perm_idx = perm_idx.to(device)

                outputs = model(tiles)
                loss = criterion(outputs, perm_idx)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += perm_idx.size(0)
                correct += (predicted == perm_idx).sum().item()

        val_accuracy = 100 * correct / total
        avg_val_loss = val_loss / len(valid_loader)
        logging.info(f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_accuracy': best_accuracy,
            'per_epoch_loss': per_epoch_loss
        }

        # Save last checkpoint
        torch.save(checkpoint, last_checkpoint_path)

        # Save best checkpoint if current validation accuracy is better
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            logging.info('Saving best checkpoint')
            # Create the checkpoint with the updated best_accuracy
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_accuracy,
                'per_epoch_loss': per_epoch_loss
            }
            torch.save(checkpoint, best_checkpoint_path)

    logging.info('Training completed!')

    return {'best_checkpoint_accuracy': best_accuracy, 'per_epoch_loss': per_epoch_loss, 'config': config}
'''
Script with functions to train Rotation
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

from rotation import RotationDataset, RotationNet 

from data_utils import reduce_dataset #TODO

def run_training_rotation(config):
    '''
    Runs a training loop for the Rotation PTT
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
    dataset = load_dataset(config['ptt_dataset_name'], token=config['hf_access_token'], cache_dir=download_path)

    if 'training_data_proportion' in config:  ## reduce training data (for imagenet1k)
        logging.info(f"Reducing training data to {config['training_data_proportion']}...")
        if config['training_data_proportion'] < 1.0:
            dataset = reduce_dataset(
                dataset,
                proportion=config['training_data_proportion'],
                seed=config['seed'])

    model = RotationNet(n_rotations=4, architecture=config['rotation_architecture'])
    print('model initialized')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    transform = transforms.Compose([transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),transforms.ToTensor()])
    validation_name = 'valid' if config['ptt_dataset_name'] == 'zh-plus/tiny-imagenet' else 'validation'  # hotfix because the validation partition in imagenet is called 'valid'

    train_dataset = RotationDataset(dataset['train'], transform=transform, architecture=config['rotation_architecture'])
    valid_dataset = RotationDataset(dataset[validation_name], transform=transform, architecture=config['rotation_architecture'])

    batch_size = 64 if config['rotation_architecture'] == 'resnet' else 64
    print('batch_size:', batch_size)

    train_loader = data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    valid_loader = data.DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

    ## instance JigsSaw model
    model = RotationNet(
        n_rotations = 4, 
        architecture = config['rotation_architecture'],
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
        for batch_idx, (rotated_imgs, rotation_labels) in enumerate(tqdm(train_loader)):
            rotated_imgs = rotated_imgs.view(-1, 3, 224, 224).to(device)  # Shape: [batch_size * 4, 3, 224, 224]
            rotation_labels = rotation_labels.view(-1).to(device)  # Shape: [batch_size * 4]

            optimizer.zero_grad()
            outputs = model(rotated_imgs)  # Shape: [batch_size * 4, n_rotations]
            loss = criterion(outputs, rotation_labels)  # Shape: [batch_size * 4]
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
            for rotated_imgs, rotation_labels in tqdm(valid_loader):
                rotated_imgs = rotated_imgs.view(-1, 3, 224, 224).to(device)  # Shape: [batch_size * 4, 3, 255, 255]
                rotation_labels = rotation_labels.view(-1).to(device)  # Shape: [batch_size * 4]

                outputs = model(rotated_imgs)  # Shape: [batch_size * 4, n_rotations]
                loss = criterion(outputs, rotation_labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += rotation_labels.size(0)
                correct += (predicted == rotation_labels).sum().item()

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
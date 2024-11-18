'''
Linear probe (logistic regression)

'''

import pytorch_lightning as pl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from datasets import load_dataset
import random
import datasets
from pathlib import Path
import numpy as np
from torchvision import transforms, models
import torch.utils.data as data
from tqdm.auto import tqdm
import pytorch_lightning as pl
import torch
import torch
import torchvision.models as models
import torch.nn as nn

from datasets import DatasetDict

import yaml
import argparse
import os
import json

import sys
sys.path.append('/home/emilio.villa/nlp_playground/CV_PTTs/jigsaw')

from jigsaw import JigsawNet

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run experiments with configuration.")
    parser.add_argument(
        '--config', 
        type=str, 
        required=True, 
        help="Path to the YAML configuration file"
    )
    parser.add_argument(
        '--log_dir', 
        type=str, 
        default='./logs', 
        help="Directory to save logs (default: ./logs)"
    )
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    print(f"Configuration loaded from {config_path}")
    return config

def reduce_dataset_to_k_labels(dataset, K, validation_split_name):
    # Get the unique labels from the training set
    train_labels = list(set(dataset["train"]["label"]))
    
    selected_labels = random.sample(train_labels, K)
    selected_labels_set = set(selected_labels)
    train_labels_array = np.array(dataset["train"]["label"])
    valid_labels_array = np.array(dataset[validation_split_name]["label"])

    train_indices = np.where(np.isin(train_labels_array, selected_labels))[0]
    valid_indices = np.where(np.isin(valid_labels_array, selected_labels))[0]

    filtered_train = dataset["train"].select(train_indices.tolist())
    filtered_valid = dataset[validation_split_name].select(valid_indices.tolist())

    filtered_dataset = {
        "train": filtered_train,
        "valid": filtered_valid
    }
    return filtered_dataset

def reduce_dataset_to_p(dataset_dict, proportion, seed=42):
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

class ClassificationDataset(data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image'].convert('RGB')
        label = self.dataset[idx]['label']
        if self.transform:
            image = self.transform(image)
        return image, label

class ResNetFeatureExtractor(nn.Module):
    '''
    Given a pretrained ResNet model, extract features up to layer N [0,4]
    
    TODO -> update for resnet50
    '''
    def __init__(self,
                 N,
                 pretrained_model = None ## this should be a resnet model
                ):
        
        super(ResNetFeatureExtractor, self).__init__()
        self.N = N

        if pretrained_model is None:
            ## instance resnet from scratch
            print('resnet not provided, using random initialization')
            # pretrained_model = models.resnet18(weights=None) ##
            pretrained_model = models.resnet50(weights=None) ##
        else:
            ## instance using previously trained resnet
            print('resnet model provided, using it to initialize features')
            # self.pretrained_model = pretrained_model

        layers = [
            pretrained_model.conv1,
            pretrained_model.bn1,
            pretrained_model.relu,
            pretrained_model.maxpool
        ]

        if N >= 1:
            layers.append(pretrained_model.layer1)
        if N >= 2:
            layers.append(pretrained_model.layer2)
        if N >= 3:
            layers.append(pretrained_model.layer3)
        if N >= 4:
            layers.append(pretrained_model.layer4)

        self.features = nn.Sequential(*layers)
        self.avgpool = pretrained_model.avgpool

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

def extract_features(
    image_dataset,
    feature_extraction_model,
    batch_size = 512,
    device = 'cuda'):
    '''
    Generates numpy array with features given a pretrained resnet model.
    Parameters:
    - image_dataset : Dataset object with the images to extract features from 
        NOTE: this is for the moment teh <ClassificationDataset> which generates images and labels, 
        should use other that only returns images
    - feature_extraction_model : Module to extract features
    - batch_size
    Returns
    - features : Numpy array with extracted features
    '''
    loader = data.DataLoader(
            image_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
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

def save_dict_to_file(my_dict, experiment_name, folder = None):
    savename = f'{experiment_name}.txt' if folder is None else f'{folder}/{experiment_name}.txt'
    with open(savename, 'w') as f:
        f.write(f'{experiment_name}\n')
        for key, value in my_dict.items():
            f.write(f"{key}\t{value}\n")

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

## Load and preprocess dataset
hub_names_dict = {
    'imagenet1k': 'ILSVRC/imagenet-1k',
    'tiny_imagenet' : 'zh-plus/tiny-imagenet',
    'voc2007' : 'clip-benchmark/wds_voc2007',
    'caltech101': 'clip-benchmark/wds_vtab-caltech101',
    
}

validation_split_dict = { #since the validation split is named different in each dataset we use this to standardize
    'imagenet1k': 'validation',
    'tiny_imagenet' : 'valid',
    'voc2007' : 'test',
    'caltech101': 'test',
}

def run_linear_probe(config):
    '''
    Runs an experiment using a linear probe.
    '''
    ##
    pl.seed_everything(42, workers=True)

    classification_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    
    dataset = load_dataset(hub_names_dict[config['dataset']])

    if 'cls' in dataset['train'].features:
        dataset = transform_dataset_clip_benchmark(dataset)

    print(f'running evaluation on {hub_names_dict[config["dataset"]]}')

    ### instance model class to load checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    ### load checkpoint from trained data
    if config['pretrained_path'] is not None:
        print('pretrained path provided ... loading backbone')
        backbone = models.resnet50(weights=None)
        backbone.fc = nn.Identity()
        tmp  = torch.load(config['pretrained_path'], map_location=torch.device('cpu'))
        backbone.load_state_dict(tmp)
    else:
        backbone = None

    metrics = {}

    if config.get('p_reduction',1.0) < 1.0:

        print('Reducing dataset to {}'.format(config['p_reduction']))
        dataset = reduce_dataset_to_p(dataset, config['p_reduction'], seed=42)

    # check if folder output exists
    output_folder = config.get('output_folder','probe_output')
    os.makedirs(output_folder, exist_ok=True)
    print('will save in {}'.format(output_folder))

    for N in config['N_values']:

        feature_extractor = ResNetFeatureExtractor(N=N, pretrained_model = backbone).to(device)

        print(f'Running N:{N}')
        
        train_classification_dataset = ClassificationDataset(dataset['train'], transform = classification_transform)
        val_classification_dataset = ClassificationDataset(dataset[validation_split_dict[config['dataset']]], transform = classification_transform)

        X_train, y_train = extract_features(
            image_dataset = train_classification_dataset,
            feature_extraction_model = feature_extractor,
            batch_size = 1024,
            device = device
        )
        X_test, y_test = extract_features(
            image_dataset = val_classification_dataset,
            feature_extraction_model = feature_extractor,
            batch_size = 1024,
            device = device
        )

        # store extracted features -> could be useful later
        np.savez_compressed(os.path.join(output_folder, '{}_features_train_N{}.npz'.format(config['experiment_savename'],N)), X=X_train, y=y_train)
        np.savez_compressed(os.path.join(output_folder, '{}_features_test_N{}.npz'.format(config['experiment_savename'],N)), X=X_test, y=y_test)

        ## use the features to train and evaluate a linear classifier\

        scaler = StandardScaler()
        clf_logreg = LogisticRegression(max_iter=config.get('lr_max_iterations',1000))

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf_logreg.fit(X_train_scaled, y_train)

        y_pred = clf_logreg.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        
        ## get per-class performance report -> useful for some interpretability later

        report = classification_report(y_test, y_pred, output_dict=True)
        with open(os.path.join(output_folder, '{}_classification_report_N{}.json'.format(config['experiment_savename'],N)), 'w') as f:
            json.dump(report, f, indent=4)

        ## somehow report
        metrics[f'{N}'] = accuracy
    
    save_dict_to_file(metrics, config['experiment_savename'], folder=output_folder)

if __name__ == "__main__":
    args = parse_args()
    
    # Load the configuration file
    config = load_config(args.config)
    
    # Run the experiment
    results = run_linear_probe(config)

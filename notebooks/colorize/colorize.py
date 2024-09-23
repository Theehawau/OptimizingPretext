import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms, models
import torchvision.transforms as T
from skimage.color import rgb2lab, rgb2gray,lab2rgb
from PIL import Image
import numpy as np
import os
import random
from tqdm import tqdm
from datasets import load_dataset
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

tinyImageNet_dataset = load_dataset("zh-plus/tiny-imagenet", cache_dir="datasets/")


class ColorizeData(data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        """
        Input:
            hf_dataset: HuggingFace Dataset object.
            transform: Optional transform to be applied on a sample.
        """
        self.dataset = hf_dataset
        self.transform = transform or T.Compose([T.Resize(256)])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
       
        # Load image from the HuggingFace dataset and convert to RGB
        input = self.dataset[index]['image'].convert('RGB') 
        input = self.transform(input) # Apply data augmentation/normalization using the self.transform function
        input = np.asarray(input) # Convert the input data to a numpy array
        img_lab = rgb2lab(input) # Convert the input image from RGB to LAB color space
        img_lab = (img_lab + 128) / 255 # Normalize the LAB values so that they are in the range of 0 to 1
        img_ab = img_lab[:, :, 1:3] # Get the "ab" channels from the LAB image
        img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float() # Convert the "ab" channels to a PyTorch tensor
        input = rgb2gray(input) # Convert the input image from RGB to grayscale
        input = torch.from_numpy(input).unsqueeze(0).float() # Convert the grayscale image to a PyTorch tensor and add a batch dimension
        return input, img_ab # Return the grayscale image and the "ab" channels as a tuple


class ColorizeNet(nn.Module):
  def __init__(self, input_size=512, front_end="resnet"):
    super(ColorizeNet, self).__init__()
    if front_end == "resnet":
        resnet = models.resnet50()
        resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1))  # Convert resnet input from 3 to 1 channel
        self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0:6])
        RESNET_FEATURE_SIZE = 512
    elif front_end == "vgg":
        raise NotImplementedError("VGG not implemented yet")
    # Upsampling Network
    self.upsample = nn.Sequential(     
      nn.Conv2d(RESNET_FEATURE_SIZE, 256, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(),
      nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(16),
      nn.ReLU(),
      nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1), # Output has 2 channels instead of 3 because we are
                                                          # predicting a and b channels of Lab color space instead of RGB
      nn.Upsample(scale_factor=2)
    )

  def forward(self, input):
    midlevel_features = self.midlevel_resnet(input)
    output = self.upsample(midlevel_features)
    return output    

def to_rgb(grayscale_input, ab_input, save_path=None, save_name=None):
      # Show/save rgb image from grayscale and ab channels
      os.makedirs(save_path['grayscale'], exist_ok=True)
      os.makedirs(save_path['colorized'], exist_ok=True)
      plt.clf() # clear matplotlib 
      color_image = torch.cat((grayscale_input, ab_input), 0).numpy() # combine channels
      color_image = color_image.transpose((1, 2, 0))  # rescale for matplotlib
      color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
      color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128   
      color_image = lab2rgb(color_image.astype(np.float64))

      grayscale_input = grayscale_input.squeeze().numpy()
      if save_path is not None and save_name is not None: 
        plt.imsave(arr=grayscale_input, fname='{}{}'.format(save_path['grayscale'], save_name), cmap='gray')
        plt.imsave(arr=color_image, fname='{}{}'.format(save_path['colorized'], save_name))

if __name__ == '__main__':
    # dataset = ColorizeData(tinyImageNet_dataset['train'])
    # dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True)
    # for i, (input, target) in enumerate(dataloader):
    #     print(input.shape, target.shape)
    #     break
    
    print('Loading the model')
    model = ColorizeNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)    
    print('Model loaded to device {}'.format(device))
    
    # Create the datasets and dataloaders
    train_dataset = ColorizeData(tinyImageNet_dataset['train'])
    valid_dataset = ColorizeData(tinyImageNet_dataset['valid'])

    train_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=12)
    valid_loader = data.DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=12)
    
    print('Datasets and dataloaders created')
    print('Shape of dataset input: {}'.format(next(iter(train_loader))[0].shape))
    print('Shape of dataset target: {}'.format(next(iter(train_loader))[1].shape))
    
    
    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    es_patience = 30
    
    num_epochs = 100
    log_each = 100
    save_images = True
    best_val_loss = torch.tensor(float('inf'))
    
    for epoch in tqdm(range(num_epochs), leave=False, total=num_epochs):
        model.train()
        avg_loss = 0.0
        tq_obj = tqdm(train_loader, leave=True, total=len(train_loader)//128, desc=f'Train Epoch {epoch}')
        for batch_idx, (input, output) in enumerate(tq_obj):
            input, output = input.to(device), output.to(device)
            optimizer.zero_grad()
            pred = model(input)
            loss = criterion(pred, output)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            if batch_idx % log_each == 0:
                # print(f'Epoch: {epoch} Batch: {batch_idx} Loss: {avg_loss:.4f}')
                tq_obj.set_postfix({'loss': avg_loss})
                avg_loss = 0.0
        scheduler.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            already_saved_images = False
            for batch_idx, (input, output) in enumerate(valid_loader):
                input, output = input.to(device), output.to(device)
                pred = model(input)
                loss = criterion(pred, output)
                val_loss += loss.item()
            if save_images and not already_saved_images:
                already_saved_images = True
                for j in range(min(len(output), 5)): # save 10 images each epoch
                    save_path = {'grayscale': 'outputs/gray/', 'colorized': 'outputs/color/', 'ground_truth': 'outputs/ground_truth/'}
                    save_name = 'img-{}-epoch-{}.jpg'.format(batch_idx * valid_loader.batch_size + j, epoch+1)
                    to_rgb(grayscale_input=input[j].cpu(), ab_input=pred[j].detach().cpu(), save_path=save_path, save_name=save_name)

        print(f'Epoch: {epoch} Validation Loss: {val_loss/len(valid_loader):.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f'Best model saved with loss: {best_val_loss}')
            torch.save(model, f'weights/colorize_resnet_best.pth')
        torch.save(model, f'weights/colorize_resnet_last.pth')
            
    
    
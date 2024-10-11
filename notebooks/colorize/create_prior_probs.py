import torch
import numpy as np
from skimage.io import imread
from skimage import color
import sys
import random
from skimage.transform import resize
from datasets import load_dataset

# Load the list of images and points
with open('data/train.txt') as lists_f:
    filename_lists = [img_f.strip() for img_f in lists_f]
    
image_paths = load_dataset("zh-plus/tiny-imagenet", cache_dir="datasets/")['train']


points = np.load('resources/pts_in_hull.npy').astype(np.float64)
points = torch.tensor(points, dtype=torch.float64)  # Convert to PyTorch tensor
points = points.unsqueeze(0)  # Add an extra dimension

# # Shuffle filenames
# random.shuffle(filename_lists)

# Initialize probability array
probs = np.zeros((313), dtype=np.float64)
num = 0

# Initialize PyTorch device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
points = points.to(device)  # Move points to the appropriate device

for img_f in filename_lists:
    img_f = img_f.strip()
    img = imread(img_f)
    img = resize(img, (224, 224), preserve_range=True)

    # Ensure the image is valid with 3 color channels
    if len(img.shape) != 3 or img.shape[2] != 3:
        continue

    # Convert image to Lab color space
    img_lab = color.rgb2lab(img)
    img_lab = img_lab.reshape((-1, 3))
    img_ab = img_lab[:, 1:]

    # Convert img_ab to PyTorch tensor and move to device
    img_ab_tensor = torch.tensor(img_ab, dtype=torch.float64).to(device)

    # Calculate distances between img_ab and points
    expanded_img_ab = img_ab_tensor.unsqueeze(1)  # Add a dimension for broadcasting
    distance = torch.sum((expanded_img_ab - points) ** 2, dim=2)

    # Find the index of the minimum distance
    nd_index = torch.argmin(distance, dim=1).cpu().numpy()  # Move result back to CPU

    # Update probabilities
    for i in nd_index:
        probs[int(i)] += 1

    print(num)
    sys.stdout.flush()
    num += 1

# Normalize probabilities
probs = probs / np.sum(probs)

# Save probabilities to file
np.save('probs', probs)

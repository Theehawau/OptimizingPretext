import torch, torchvision
import random
from optimizepretext.data.tinyimagenet import df
from optimizepretext.utils.misc import Memory
from optimizepretext.models.jigsaw import JigsawNetwork

class JigsawLoader():
    __acceptable_params = [ 'cfg', 'split']
    def __init__(self, **kwargs):
        [setattr(self, k, kwargs.get(k, None)) for k in self.__acceptable_params]
        super(JigsawLoader, self).__init__()
        self.df = df[self.split]
        self.color_transform = torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4)
        self.flips = [torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomVerticalFlip()]
        self.normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        original = self.df[index]['image']
        # Some images are in L mode, we need to convert them to RGB
        if original.mode != 'RGB':
            original = original.convert('RGB')
       
        crop_dim = 22

        crop_areas = [(i*crop_dim, j*crop_dim, (i+1)*crop_dim, (j+1)*crop_dim) for i in range(3) for j in range(3)]
        samples = [original.crop(crop_area) for crop_area in crop_areas]
        samples = [torchvision.transforms.RandomCrop((21, 21))(patch) for patch in samples]
        # augmentation collor jitter
        image = self.color_transform(original)
        
        samples = [self.color_transform(patch) for patch in samples]
        # augmentation - flips
        image = self.flips[0](image)
        image = self.flips[1](image)
        # to tensor
        image = torchvision.transforms.functional.to_tensor(image)
        samples = [torchvision.transforms.functional.to_tensor(patch) for patch in samples]
        # normalize
        image = self.normalize(image)
        samples = [self.normalize(patch) for patch in samples]
        random.shuffle(samples)

        return {'original': image, 'patches': samples, 'index': index}

if __name__ == '__main__':
    from torchvision.models.resnet import resnet50
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = JigsawLoader(split='train')
    
    print(len(dataset))
    train_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=32, num_workers=12)
    memory = Memory(size=len(dataset), weight=0.5, device=device)
    kwargs = {
        'cfg': None,
        'front_end': resnet50
        }
    net = JigsawNetwork(**kwargs).to(device)
    # x = torch.rand(2,3, 64,64)
    memory.initialize(net, train_loader)  
    for step, batch in enumerate(train_loader):
        # prepare batch
        images = batch['original'].to(device)
        patches = [element.to(device) for element in batch['patches']]
        index = batch['index']
        representations = memory.return_representations(index).to(device).detach()
        breakpoint()      
        
    

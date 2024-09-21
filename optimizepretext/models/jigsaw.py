import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50


class JigsawNetwork(nn.Module):
    def __init__(self, cfg, front_end):
        super(JigsawNetwork, self).__init__()
        # self.network = resnet50()
        self.network = front_end()
        self.network = torch.nn.Sequential(*list(self.network.children())[:-1])
        self.projection_original_features = nn.Linear(2048, 128)
        self.connect_patches_feature = nn.Linear(1152, 128)

    def forward_once(self, x):
        return self.network(x)

    def return_reduced_image_features(self, original):
        original_features = self.forward_once(original)
        original_features = original_features.view(-1, 2048)
        original_features = self.projection_original_features(original_features)
        return original_features

    def return_reduced_image_patches_features(self, original, patches):
        original_features = self.return_reduced_image_features(original)

        patches_features = []
        for i, patch in enumerate(patches):
            patch_features = self.return_reduced_image_features(patch)
            patches_features.append(patch_features)

        patches_features = torch.cat(patches_features, axis=1)

        patches_features = self.connect_patches_feature(patches_features)
        return original_features, patches_features

    def forward(self, images=None, patches=None, mode=0):
        '''
        mode 0: get 128 feature for image,
        mode 1: get 128 feature for image and patches       
        '''
        if mode == 0:
            return self.return_reduced_image_features(images)
        if mode == 1:
            return self.return_reduced_image_patches_features(images, patches)


if __name__ == "__main__":
    kwargs = {
        'cfg': None,
        'front_end': resnet50
        }
    model = JigsawNetwork(**kwargs)
    x = torch.rand(2,3, 64,64)
    out = model(x, mode=0)
    print(out.shape)
    print(model)
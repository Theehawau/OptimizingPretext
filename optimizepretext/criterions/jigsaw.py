'''
Adapted from: https://github.com/akwasigroch/Pretext-Invariant-Representations/
'''

import torch

class NoiseContrastiveEstimator():
    def __init__(self, device):
        self.device = device

    def __call__(self, original_features, path_features, index, memory, negative_nb=1000):
        loss = 0
        for i in range(original_features.shape[0]):

            temp = 0.07
            cos = torch.nn.CosineSimilarity()
            criterion = torch.nn.CrossEntropyLoss()

            negative = memory.return_random(size=negative_nb, index=[index[i]])
            negative = torch.Tensor(negative).to(self.device).detach()

            image_to_modification_similarity = cos(original_features[None, i, :], path_features[None, i, :])/temp
            matrix_of_similarity = cos(path_features[None, i, :], negative) / temp

            similarities = torch.cat((image_to_modification_similarity, matrix_of_similarity))
            loss += criterion(similarities[None, :], torch.tensor([0]).to(self.device))
        return loss / original_features.shape[0]
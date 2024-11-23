import random
from datasets import DatasetDict

def reduce_dataset(dataset_dict, proportion, seed=42):
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
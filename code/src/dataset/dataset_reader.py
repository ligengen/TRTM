import torch
from torch.utils.data import Dataset


class DatasetReader(Dataset):
    """
    Abstract class.
    """

    def __init__(self, dataset_name, dataset_path, data_transforms=None):
        self.dataset_name = dataset_name
        self.data_transforms = data_transforms
        self.dataset_path = dataset_path

    def __len__(self):
        raise NotImplementedError("Must be implemented by subclass")

    def load_sample(self, idx):
        raise NotImplementedError("Must be implemented by subclass")

    def __getitem__(self, idx):
        sample = self.load_sample(idx)

        if self.data_transforms:
            # Apply data transformation
            sample = self.data_transforms(sample)

        return sample

import torch
import torchvision
import os
import pickle as pkl
import numpy as np

import torch.utils.data as torch_data
from torchvision import transforms


def get_input_transform():
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.float()),
        transforms.Lambda(lambda x: x.permute(2, 0, 1)),
        transforms.Lambda(lambda x: x.div_(255)),
    ])
    return transform


def get_label_transform():
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.long()),
    ])
    return transform


class DatasetOfPartitions(torch_data.Dataset):
    def __init__(self, input_dir, label_dir=None, ext='npy', func_load=None):
        """
        Make a dataset that return iterators which iterate over sub_datasets themselves.
        The data for each sub_dataset is in seperate files
        """
        super(DatasetOfPartitions).__init__()

        self.has_labels = True if label_dir else False
        self.input_files = sorted([os.path.join(input_dir, x) for x in os.listdir(input_dir) \
                                   if x.endswith(ext)])

        if self.has_labels:
            self.label_files = sorted([os.path.join(label_dir, x) for x in os.listdir(label_dir) \
                                       if x.endswith(ext)])
            assert len(self.input_files) == len(self.label_files)

        self.file_indices = range(len(self.input_files))

        if ext == 'pkl':
            self.func_load = pkl.load
        elif ext == 'npy':
            self.func_load = np.load
        else:
            assert func_load is not None

    def __getitem__(self, index):
        partition_id = self.file_indices[index]

        input = self.func_load(self.input_files[partition_id])

        if self.has_labels:
            label = self.func_load(self.label_files[partition_id])
            return [torch.from_numpy(input), torch.from_numpy(label)]
        else:
            return [torch.from_numpy(input)]

    def __len__(self):
        return len(self.file_indices)


class CustomDataset(torch_data.Dataset):
    def __init__(self, input_data, label_data=None):

        if label_data is not None:
            self.has_labels = True
        else:
            self.has_labels = False
        self.transform_input = get_input_transform()
        self.transform_label = get_label_transform()

        self.input_data = input_data
        if self.has_labels:
            self.label_data = label_data

    def __getitem__(self, index):

        raw_input = self.input_data[index]
        input = self.transform_input(raw_input)

        if self.has_labels:
            label = self.transform_label(self.label_data[index])
            return (input, label)
        else:
            return input

    def __len__(self):
        return self.input_data.shape[0]

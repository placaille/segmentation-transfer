import torch
import h5py
import torchvision

import torch.utils.data as torch_data
from torchvision import transforms


def get_transform():
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.permute(2, 0, 1)),
        transforms.Lambda(lambda x: x.div_(255)),
    ])
    return transform


class Hdf5TorchDatasetWithLabels(torch_data.Dataset):
    def __init__(self, input_path, label_path, dataset_name):
        super(Hdf5TorchDatasetWithLabels).__init__()

        input_file = h5py.File(input_path, mode='r')
        label_file = h5py.File(label_path, mode='r')

        self.transform_input = get_transform()

        self.input_data = input_file[dataset_name]
        self.label_data = label_file[dataset_name]

    def __getitem__(self, index):
        input = torch.from_numpy(self.input_data[index]).float()
        label = torch.from_numpy(self.label_data[index]).int()
        return self.transform_input(input), label

    def __len__(self):
        return self.input_data.shape[0]


class Hdf5TorchDatasetWithoutLabels(torch_data.Dataset):
    def __init__(self, input_path, dataset_name):
        super(Hdf5TorchDatasetWithoutLabels).__init__()

        input_file = h5py.File(input_path, mode='r')
        self.transform_input = get_transform()
        self.input_data = input_file[dataset_name]

    def __getitem__(self, index):
        input = torch.from_numpy(self.input_data[index]).float()
        return self.transform_input(input)

    def __len__(self):
        return self.input_data.shape[0]

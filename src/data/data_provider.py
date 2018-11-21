import click
import torch
import math
import pickle as pkl
import numpy as np

from torch.utils.data import DataLoader, TensorDataset


def tensor_dataset_to_cuda(tensor_dataset, device):
    inputs = tensor_dataset.tensors[0].to(device)
    labels = tensor_dataset.tensors[1].to(device)
    return TensorDataset(inputs, labels)


class DataProvider(object):
    """
    DataProvider to create train/valid/test iterators
    Arguments:
        dataset_file (str): Path to preprocessed TensorDataset (train,valid,test)
        batch_size (int): Batch size
    """
    def __init__(self, dataset_file, batch_size, device):

        with open(dataset_file, 'rb') as file:
            train, valid, test = pkl.load(file)

        # load on gpu if applicable
        train = tensor_dataset_to_cuda(train, device)
        valid = tensor_dataset_to_cuda(valid, device)
        test = tensor_dataset_to_cuda(test, device)

        self.num_train = len(train)
        self.num_valid = len(valid)
        self.num_test = len(test)

        self.num_classes = train.tensors[1].max().item() + 1
        self.num_pixels = np.prod(train.tensors[0].shape[1:])
        self.input_channels, self.input_width, self.input_height = train.tensors[0].shape[1:]
        self.iters_per_epoch = math.ceil(self.num_train / batch_size)

        self.train_iterator = DataLoader(train, batch_size, shuffle=True)
        self.valid_iterator = DataLoader(valid, batch_size)
        self.test_iterator = DataLoader(test, batch_size)

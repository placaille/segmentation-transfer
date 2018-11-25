import click
import torch
import math
import numpy as np

from torch.utils.data import DataLoader
from data.dataset import Hdf5TorchDatasetWithLabels, Hdf5TorchDatasetWithoutLabels


class Hdf5DataProvider(object):
    def __init__(self, input_hdf5_path, batch_size, num_classes, label_hdf5_path=None,
                 num_workers=0):
        """
        DataProvider to create train/valid/test iterators
        Arguments:
            input_hdf5_path (str): path to hdf5 with 'train' and 'valid' datasets
            batch_size  (int)
            num_classes (int)
            label_hdf5_path (str): default None
        """
        if label_hdf5_path is None:
            train_dataset = Hdf5TorchDatasetWithoutLabels(input_hdf5_path, 'train')
            valid_dataset = Hdf5TorchDatasetWithoutLabels(input_hdf5_path, 'valid')
        else:
            train_dataset = Hdf5TorchDatasetWithLabels(input_hdf5_path, label_hdf5_path, 'train')
            valid_dataset = Hdf5TorchDatasetWithLabels(input_hdf5_path, label_hdf5_path, 'valid')

        self.num_train = len(train_dataset)
        self.num_valid = len(valid_dataset)
        self.num_classes = num_classes

        # input data in hfg5 is (N, 120, 160, 3)
        self.num_pixels = np.prod(train_dataset.input_data.shape[1:])
        self.input_channels = train_dataset.input_data.shape[-1]
        self.iters_per_epoch = math.ceil(self.num_train / batch_size)

        self.train_iterator = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
        self.valid_iterator = DataLoader(valid_dataset, batch_size, num_workers=num_workers)

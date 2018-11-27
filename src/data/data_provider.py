import click
import torch
import math
import os
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from data.dataset import DatasetOfPartitions, CustomDataset


class PartitionProvider(object):
    def __init__(self, input_dir, label_dir=None, num_workers=0,
                 partition_batch_size=32, partition_num_workers=0):
        """
        Provider for partitions of dataset which is split in files
        """

        input_train_dir = os.path.join(input_dir, 'train')
        input_valid_dir = os.path.join(input_dir, 'valid')

        if label_dir:
            label_train_dir = os.path.join(label_dir, 'train')
            label_valid_dir = os.path.join(label_dir, 'valid')
        else:
            label_train_dir = None
            label_valid_dir = None

        train_partitions = DatasetOfPartitions(input_train_dir, label_train_dir)
        valid_partitions = DatasetOfPartitions(input_valid_dir, label_valid_dir)

        self.num_train_files = len(train_partitions)
        self.num_valid_files = len(valid_partitions)
        self.partition_batch_size = partition_batch_size
        self.partition_num_workers = partition_num_workers

        self.train_partition_iterator = DataLoader(
            train_partitions,
            batch_size=1,
            shuffle=True,
            num_workers=num_workers
        )
        self.valid_partition_iterator = DataLoader(
            valid_partitions,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers
        )

    def get_train_iterator(self, partition):
        if len(partition) == 2:
            input_partition, label_partition = partition
            dataset = CustomDataset(input_partition[0], label_partition[0])
        elif len(partition) == 1:
            dataset = CustomDataset(partition[0])

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.partition_batch_size,
            shuffle=True,
            num_workers=self.partition_num_workers
        )
        return data_loader

    def get_valid_iterator(self, partition):
        if len(partition) == 2:
            input_partition, label_partition = partition
            dataset = CustomDataset(input_partition[0], label_partition[0])
        elif len(partition) == 1:
            dataset = CustomDataset(partition[0])

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.partition_batch_size,
            shuffle=False,
            num_workers=self.partition_num_workers
            )
        return data_loader

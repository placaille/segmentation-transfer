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
            num_workers=self.partition_num_workers,
            drop_last=True
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


class InfiniteProviderFromPartitions(object):
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
        self.num_workers = num_workers
        self.partition_batch_size = partition_batch_size
        self.partition_num_workers = partition_num_workers

        self.train_partition_loader = DataLoader(
            train_partitions,
            batch_size=1,
            shuffle=True,
            num_workers=num_workers
        )
        self.valid_partition_loader = DataLoader(
            valid_partitions,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers
        )


    def init_iterator(self, train=True):
        self.train = train
        if self.train:
            partition_loader = self.train_partition_loader
            get_loader = self.get_train_loader
        else:
            partition_loader = self.valid_partition_loader
            get_loader = self.get_valid_loader

        # init partition and get first
        partition_iterator = iter(partition_loader)
        partition = next(partition_iterator)

        # get batch iterator based on partition
        iterator = iter(get_loader(partition))
        if self.train:
            self.train_partition_iterator = partition_iterator
            self.train_iterator = iterator
        else:
            self.valid_partition_iterator = partition_iterator
            self.valid_iterator = iterator


    def __next__(self):
        if self.train:
            partition_iterator = self.train_partition_iterator
            partition_loader = self.train_partition_loader
            get_loader = self.get_train_loader
            iterator = self.train_iterator
        else:
            partition_iterator = self.valid_partition_iterator
            partition_loader = self.valid_partition_loader
            get_loader = self.get_valid_loader
            iterator = self.valid_iterator

        try:
            batch = next(iterator)
        except StopIteration:
            try:
                # get next partition
                partition = next(partition_iterator)
            except StopIteration:
                # restart partition_iterator
                partition_iterator = iter(partition_loader)
                partition = next(partition_iterator)

            # get batch iterator based on partition
            iterator = iter(get_loader(partition))
            batch = next(iterator)

        if self.train:
            self.train_partition_iterator = partition_iterator
            self.train_partition_loader = partition_loader
            self.train_iterator = iterator
        else:
            self.valid_partition_iterator = partition_iterator
            self.valid_partition_loader = partition_loader
            self.valid_iterator = iterator

        return batch


    def get_train_loader(self, partition):
        if len(partition) == 2:
            input_partition, label_partition = partition
            dataset = CustomDataset(input_partition[0], label_partition[0])
        elif len(partition) == 1:
            dataset = CustomDataset(partition[0][0])

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.partition_batch_size,
            shuffle=True,
            num_workers=self.partition_num_workers
        )
        return data_loader

    def get_valid_loader(self, partition):
        if len(partition) == 2:
            input_partition, label_partition = partition
            dataset = CustomDataset(input_partition[0], label_partition[0])
        elif len(partition) == 1:
            dataset = CustomDataset(partition[0][0])

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.partition_batch_size,
            shuffle=False,
            num_workers=self.partition_num_workers
            )
        return data_loader

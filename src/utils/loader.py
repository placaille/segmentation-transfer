import torch
import h5py

class H5PYDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, classes, transform=None):
        super(H5PYDataset, self).__init__()
        self.tensor = dataset
        self.classes = classes

    def __getitem__(self, index):
        input = self.transform(self.tensor[index]) if self.transform is not None else self.tensor[index]
        return input, self.classes[index]

    def __len__(self):
        return len(self.tensor)

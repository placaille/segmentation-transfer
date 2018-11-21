import click
import os
import torch
import pickle as pkl
import numpy as np

from PIL import Image
from torchvision import transforms


def get_transform():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    return transform


def convert_to_dataset(input_data, label_data, idx):
    input_split = torch.FloatTensor(input_data[idx])
    label_split = torch.LongTensor(label_data[idx])

    return torch.utils.data.TensorDataset(input_split, label_split)


def read_raw_data(file_name):
    with open(file_name, 'rb') as file:
        data = np.load(file_name)
    return data


@click.command()
@click.argument('image_dir', type=click.Path(exists=True, readable=True, file_okay=False))
@click.argument('output_file', type=click.Path(writable=True, dir_okay=False))
def process_images(image_dir, output_file):
    """
    Use if have a directory of images
    """
    print('Preprocessing data..')

    transform = get_transform()

    images = []
    valid_ext = ['jpg', 'png']
    for file in os.listdir(image_dir):

        is_image = True in [file.endswith(ext) for ext in valid_ext]
        if not is_image:
            continue

        image_file = os.path.join(image_dir, file)
        images.append(transform(Image.open(image_file)))

    images = torch.stack(images)

    with open(output_file, 'wb') as file:
        pkl.dump(images, file)


@click.command()
@click.argument('input_file', type=click.Path(exists=True, readable=True, dir_okay=False))
@click.argument('label_file', type=click.Path(exists=True, readable=True, dir_okay=False))
@click.argument('output_file', type=click.Path(writable=True, dir_okay=False))
@click.option('--perc_train', default=0.6, type=float, help='percentage of data for train (def .6)')
@click.option('--perc_valid', default=0.2, type=float, help='percentage of data for valid (def .2)')
@click.option('--perc_test', default=0.2, type=float, help='percentage of data for test (def .2)')
@click.option('--shuffle', is_flag=True)
def main(input_file, label_file, output_file, perc_train, perc_valid, perc_test, shuffle):
    print('Preprocessing data..')

    input_data = read_raw_data(input_file)
    label_data = read_raw_data(label_file)

    assert input_data.shape[0] == label_data.shape[0]
    assert np.isclose(np.sum([perc_train, perc_valid, perc_test]), 1.0)

    idx_all = np.arange(input_data.shape[0])
    split_train = int(np.floor(perc_train * idx_all.shape[0]))
    split_valid = int(np.floor((perc_valid / (perc_valid + perc_test)) * \
                                (idx_all.shape[0] - split_train)))

    if shuffle:
        np.random.seed(6567)
        np.random.shuffle(idx_all)

    idx_train, idx_other = idx_all[:split_train], idx_all[split_train:]
    idx_valid, idx_test = idx_other[:split_valid], idx_other[split_valid:]

    assert idx_train.shape[0] + idx_valid.shape[0] + idx_test.shape[0] == idx_all.shape[0]

    # transform = get_transform()
    # data is already normalized, just need to put it N x C x W x H
    input_data = np.transpose(input_data, (0, 3, 1, 2))

    train_dataset = convert_to_dataset(input_data, label_data, idx_train)
    valid_dataset = convert_to_dataset(input_data, label_data, idx_valid)
    test_dataset = convert_to_dataset(input_data, label_data, idx_test)

    with open(output_file, 'wb') as file:
        pkl.dump((train_dataset, valid_dataset, test_dataset), file)


if __name__ == '__main__':
    main()

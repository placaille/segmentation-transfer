import numpy as np
import imageio
import click
import torch
import os
import re

from torchvision.utils import make_grid
from torch.nn.functional import interpolate

def get_raw_grid(path_to_file):
    data = torch.tensor(np.load(path_to_file))  # (32, 3, 120, 160)
    downsampled = interpolate(data, scale_factor=0.33)
    grid = make_grid(downsampled, nrow=4, padding=2, normalize=False).numpy()
    return grid

@click.command()
@click.argument('input-dir', type=click.Path(exists=True, file_okay=False, readable=True))
@click.argument('output-file', type=click.Path(dir_okay=False, writable=True))
def main(input_dir, output_file):
    """
    used to create a gif of set of saved images under .npy format
    """
    all_files = os.listdir(input_dir)

    transformed_files = [file for file in all_files if 'img_transformed' in file]
    segmented_files = [file for file in all_files if 'img_segmented' in file]
    source_files = [file for file in all_files if 'img_source' in file]

    convert = lambda text: int(text) if text.isdigit() else text
    num_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    transformed_files = sorted(transformed_files, key=num_key)
    segmented_files = sorted(segmented_files, key=num_key)

    images = []
    source_grid = get_raw_grid(os.path.join(input_dir, source_files[0])) * 255
    for i, (transformed_file, segmented_file) in enumerate(zip(transformed_files,segmented_files)):
        if i % 50 == 0:
            print(i)

        transformed_grid = get_raw_grid(os.path.join(input_dir, transformed_file)) * 255
        segmented_grid = get_raw_grid(os.path.join(input_dir, segmented_file)) * 255

        all_tensor = torch.tensor((source_grid, transformed_grid, segmented_grid))
        meta_grid = make_grid(all_tensor, nrow=3, padding=0, normalize=False).numpy()
        images.append(np.uint8(meta_grid.transpose(1, 2, 0)))

    imageio.mimsave(output_file, images)

if __name__ == '__main__':
    main()

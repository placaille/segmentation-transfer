import click
import h5py
import numpy as np


def log_data_info(file_path, output_path, data):
    length = len(file_path)
    str = '='*length + '\n{}\n'.format(file_path) + '='*length + '\n'
    str += 'Num:\t{}\n'.format(data.shape[0])
    str += 'Shape:\t{}\n'.format(data.shape[1:])

    with open(output_path, 'a') as file:
        file.write(str)


def read_raw_data(file_name):
    with h5py.File(file_name) as file:
        data_name = list(file)[0]
        data = file[data_name][:]
    return data


@click.command()
@click.argument('input_file', type=click.Path(exists=True, readable=True, dir_okay=False))
@click.argument('output_file', type=click.Path(writable=True, dir_okay=False))
def main(input_file, output_file):

    data = read_raw_data(input_file)
    log_data_info(input_file, output_file, data)


if __name__ == '__main__':
    main()

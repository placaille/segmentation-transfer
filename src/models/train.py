import click
import sys
import torch
import os
import models
import numpy as np

sys.path.append('src')

from torch import nn, optim
from data import Hdf5DataProvider
from utils import CommandWithConfigFile, EarlyStopper
from utils import vis
from evaluate import evaluate
from itertools import count
from timeit import default_timer as timer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@click.command(cls=CommandWithConfigFile('config_file'))
@click.argument('data-sim', type=click.Path(exists=True, dir_okay=False))
@click.argument('data-real', type=click.Path(exists=True, dir_okay=False))
@click.argument('data-label', type=click.Path(exists=True, dir_okay=False))
@click.option('--save-dir', type=click.Path(writable=True, file_okay=False))
@click.option('--config-file', type=click.Path(exists=True, dir_okay=False))
@click.option('--batch-size', type=int, default=32, help='(default 32)')
@click.option('--seg-model-name', default='segnet', type=click.Choice([
              'segnet',
              ]), help='Name of model architecture (default segnet)')
@click.option('--early-stop-patience', type=int, default=10)
@click.option('--server', type=str, default='http://localhost')
@click.option('--port', type=str, default='8067')
def main(data_sim, data_real, data_label, save_dir, batch_size, config_file,
         seg_model_name, early_stop_patience, server, port):

    print('Loading data..')
    num_classes = 4
    real_data_provider = Hdf5DataProvider(data_real, batch_size, num_classes)
    import pdb;pdb.set_trace()
    sim_data_provider = Hdf5DataProvider(data_sim, batch_size, num_classes, data_label)

    print('Building model & loading on GPU (if applicable)..')
    seg_model = models.get_seg_model(seg_model_name, sim_data_provider).to(device)
    if save_dir:
        seg_model.save(os.path.join(save_dir, '{}.pth'.format(seg_model.name)))

    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(seg_model.parameters())

    print('Initializing misc..')
    batch_count = 0
    results = dict()
    early_stopper = EarlyStopper('accuracy', early_stop_patience)

    print('Starting training..')
    for epoch_id in count(start=1):
        start = timer()
        seg_model.train()
        epoch_loss = 0
        for batch_id, batch in enumerate(sim_data_provider.train_iterator):
            batch_count += 1

            input, labels = batch
            logits = seg_model(input.to(device))

            optimizer.zero_grad()
            loss_seg = loss(logits.view(-1, num_classes), labels.view(-1).to(device))
            loss_seg.backward()
            optimizer.step()

            epoch_loss += loss_seg.item()

        results['epoch_avg_loss'] = np.divide(epoch_loss, batch_id+1)
        results.update(evaluate(seg_model, sim_data_provider.valid_iterator))
        early_stopper.update(results, epoch_id)

        print('epoch {:3d} - batch {:6d} - loss {:7.4f} - valid acc {:7.4f} - {:4.1f} secs'.format(
            epoch_id, batch_count, results['epoch_avg_loss'], results['accuracy'], timer()-start
        ))
        X, data = np.array([epoch_id]), np.array([results['epoch_avg_loss']])
        visualise_plot(visualiser, X, data, title='Epoch avg loss', iteration=0, update='append')
        data = np.array([results['accuracy']])
        visualise_plot(visualiser, X, data, title='Accuracy', iteration=1, update='append')
        visualise_image(visualiser, results['images'], title='Source image', iteration=0)
        visualise_image(visualiser, results['target'], title='Target image', iteration=1)
        visualise_image(visualiser, results['segmented'], title='Segmented image', iteration=2)

        if early_stopper.new_best and save_dir:
            seg_model.save(os.path.join(save_dir, '{}.pth'.format(seg_model.name)))

        if early_stopper.stop:
            print('Stopping training due to lack of improvement..')
            print('Best performing model:\t epoch {} ({} {:7.4f})'.format(
                early_stopper.best_id,
                early_stopper.criteria,
                early_stopper.best_value))
            break

if __name__ == '__main__':
    main()

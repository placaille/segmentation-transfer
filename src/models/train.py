import click
import sys
import torch
import os
import models
import numpy as np

sys.path.append('src')

from torch import nn, optim
from data import DataProvider
from utils import CommandWithConfigFile, EarlyStopper
from evaluate import evaluate
from itertools import count
from timeit import default_timer as timer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@click.command(cls=CommandWithConfigFile('config_file'))
@click.argument('data-file', type=click.Path(exists=True, dir_okay=False))
@click.option('--save-dir', type=click.Path(writable=True, file_okay=False))
@click.option('--config-file', type=click.Path(exists=True, dir_okay=False))
@click.option('--batch-size', type=int, default=32, help='(default 32)')
@click.option('--model-name', default='segnet', type=click.Choice([
              'segnet',
              ]), help='Name of model architecture (default segnet)')
@click.option('--early-stop-patience', type=int, default=10)
def main(data_file, save_dir, batch_size, config_file, model_name,
         early_stop_patience):

    print('Loading data..')
    data_provider = DataProvider(data_file, batch_size, device)

    print('Building model & loading on GPU (if applicable)..')
    model = models.get_model(model_name, data_provider).to(device)
    if save_dir:
        model.save(os.path.join(save_dir, '{}.pth'.format(model.name)))

    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    print('Initializing misc..')
    batch_count = 0
    results = dict()
    early_stopper = EarlyStopper('accuracy', early_stop_patience)

    print('Starting training..')
    for epoch_id in count(start=1):
        start = timer()
        model.train()
        epoch_loss = 0
        for batch_id, batch in enumerate(data_provider.train_iterator):
            batch_count += 1

            input, labels = batch
            logits = model(input)

            optimizer.zero_grad()
            loss_seg = loss(logits.view(-1, data_provider.num_classes), labels.view(-1))
            loss_seg.backward()
            optimizer.step()

            epoch_loss += loss_seg.item()

        results['epoch_avg_loss'] = np.divide(epoch_loss, batch_id+1)
        results.update(evaluate(model, data_provider.valid_iterator))
        early_stopper.update(results, epoch_id)

        print('Epoch {:3d} - Batch {:6d} - loss {:7.4f} - valid acc {:7.4f} - {:4.1f} secs'.format(
            epoch_id, batch_count, results['epoch_avg_loss'], results['accuracy'], timer()-start
        ))

        if early_stopper.new_best and save_dir:
            model.save(os.path.join(save_dir, '{}.pth'.format(model.name)))

        if early_stopper.stop:
            print('Stopping training due to lack of improvement..')
            print('Best performing model:\t epoch {} ({} {:7.4f})'.format(
                early_stopper.best_id,
                early_stopper.criteria,
                early_stopper.best_value))
            break

if __name__ == '__main__':
    main()

import click
import sys
import torch
import os
import models
import numpy as np

sys.path.append('src')

from torch import nn, optim
from data import PartitionProvider
from utils import CommandWithConfigFile, EarlyStopper
from utils import vis
from evaluate import evaluate
from itertools import count
from timeit import default_timer as timer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@click.command(cls=CommandWithConfigFile('config_file'))
@click.argument('data-sim-dir', type=click.Path(exists=True, file_okay=False))
@click.argument('data-real-dir', type=click.Path(exists=True, file_okay=False))
@click.argument('data-label-dir', type=click.Path(exists=True, file_okay=False))
@click.option('--save-dir', type=click.Path(writable=True, file_okay=False))
@click.option('--visdom-dir', type=click.Path(writable=True, file_okay=False))
@click.option('--config-file', type=click.Path(exists=True, dir_okay=False))
@click.option('--batch-size', type=int, default=32, help='(default 32)')
@click.option('--seg-model-name', default='segnet', type=click.Choice([
              'segnet',
              ]), help='Name of model architecture (default segnet)')
@click.option('--early-stop-patience', type=int, default=10)
@click.option('--server', type=str, default=None)
@click.option('--port', type=str, default=None)
@click.option('--reload', is_flag=True,
               help='Flag notifying that the experiment is being reloaded.')
def main(data_sim_dir, data_real_dir, data_label_dir, save_dir, visdom_dir, batch_size, config_file,
         seg_model_name, early_stop_patience, server, port, reload):

    print('Loading data..')
    num_classes = 4
    input_channels = 3
    real_data = PartitionProvider(
        input_dir=data_real_dir,
        label_dir=None,
        num_workers=0,
        partition_batch_size=batch_size,
        partition_num_workers=2
    )

    sim_data = PartitionProvider(
        input_dir=data_sim_dir,
        label_dir=data_label_dir,
        num_workers=0,
        partition_batch_size=batch_size,
        partition_num_workers=2
    )

    print('Building model & loading on GPU (if applicable)..')
    seg_model = models.get_seg_model(seg_model_name, num_classes, input_channels).to(device)
    if save_dir:
        seg_model.save(os.path.join(save_dir, '{}.pth'.format(seg_model.name)))

    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(seg_model.parameters())

    print('Initializing misc..')
    batch_count = 0
    partition_count = 0
    results = dict()
    early_stopper = EarlyStopper('accuracy', early_stop_patience)
    visualiser = vis.Visualiser(server, port, seg_model_name, reload, visdom_dir)

    print('Starting training..')
    for epoch_id in count(start=1):
        for sim_partition in sim_data.train_partition_iterator:
            start = timer()
            seg_model.train()
            partition_loss = 0
            batch_per_part = 0
            partition_count += 1

            sim_data_train_iterator = sim_data.get_train_iterator(sim_partition)

            for batch_id, batch in enumerate(sim_data_train_iterator):
                batch_count += 1
                batch_per_part += 1

                input, labels = batch
                logits = seg_model(input.to(device))

                optimizer.zero_grad()
                loss_seg = loss(logits.view(-1, num_classes), labels.view(-1).to(device))
                loss_seg.backward()
                optimizer.step()

                partition_loss += loss_seg.item()
                X, data = np.array([batch_count]), np.array([loss_seg.detach().to('cpu').numpy()])
                visualiser.plot(X, data, title='Loss per batch', legend=['Loss'], iteration=2, update='append')

            del logits
            torch.cuda.empty_cache()

            results['partition_avg_loss'] = np.divide(partition_loss, batch_per_part)
            results.update(evaluate(seg_model, sim_data, device))
            early_stopper.update(results, epoch_id, batch_count)
            log_and_viz_results(results, epoch_id, batch_count, partition_count, visualiser, start)

            if early_stopper.new_best and save_dir:
                seg_model.save(os.path.join(save_dir, '{}.pth'.format(seg_model.name)))

            if early_stopper.stop:
                early_stopper.print_stop()
                return


def log_and_viz_results(results, epoch_id, batch_id, eval_count, visualiser, start):

    print('epoch {:3d} - batch {:6d} - loss {:7.4f} - valid acc {:7.4f} - {:4.1f} secs'.format(
        epoch_id, batch_id, results['partition_avg_loss'], results['accuracy'], timer()-start
    ))

    X, data = np.array([eval_count]), np.array([results['partition_avg_loss']])
    visualiser.plot(X, data, title='Partition avg loss', legend=['Loss'], iteration=0, update='append')
    data = np.array([results['accuracy']])
    visualiser.plot(X, data, title='Accuracy', legend=['Accuracy'], iteration=1, update='append')
    visualiser.image(results['images'], title='Source image', iteration=0)
    visualiser.image(results['targets'], title='Target image', iteration=1)
    visualiser.image(results['segmented'], title='Segmented image', iteration=2)


if __name__ == '__main__':
    main()

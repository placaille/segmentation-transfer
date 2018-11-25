±import click
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
@click.option('--run-name', type=str)
@click.option('--save-dir', type=click.Path(writable=True, file_okay=False))
@click.option('--visdom-dir', type=click.Path(writable=True, file_okay=False))
@click.option('--config-file', type=click.Path(exists=True, dir_okay=False))
@click.option('--batch-size', type=int, default=32, help='(default 32)')
@click.option('--seg-model-name', default='segnet', type=click.Choice([
              'segnet', 'segnet_strided_upsample'
              ]), help='Name of model architecture (default segnet)')
@click.option('--discr-model-name', default='dcgan_discr', help='Name of discriminator model (default dcgan_discr)',
            type=click.Choice(['dcgan_discr']))
@click.option('--gen-model-name', default='style_transfer_gen', help='Name of generator model (default style_transfer_gen)',
            type=click.Choice(['style_transfer_gen']))
@click.option('--early-stop-patience', type=int, default=10)
@click.option('--server', type=str, default=None)
@click.option('--port', type=str, default=None)
@click.option('--reload', is_flag=True,
               help='Flag notifying that the experiment is being reloaded.')
def main(data_sim_dir, data_real_dir, data_label_dir, save_dir, visdom_dir, batch_size,
         config_file, seg_model_name, discr_model_name, gen_model_name, early_stop_patience,
         server, port, reload, run_name):

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
    model_seg = models.get_seg_model(seg_model_name, num_classes, input_channels).to(device)
    model_gen = models.get_generator_model(gen_model_name, real_data_provider).to(device)
    model_discr = models.get_discriminator_model(discr_model_name, sim_data_provider).to(device)
    if save_dir:
        model_seg.save(os.path.join(save_dir, '{}.pth'.format(model_seg.name)))
        model_gen.save(os.path.join(save_dir, '{}.pth'.format(model_gen.name)))
        model_discr.save(os.path.join(save_dir, '{}.pth'.format(model_discr.name)))

    # adjusted class weights [black, white, red, yellow] see README.md
    class_weights = torch.tensor([0.0051, 0.0551, 0.6538, 0.2860]).to(device)

    obj_ce = nn.CrossEntropyLoss(weight=class_weights)
    obj_adv = nn.BCELoss()

    optim_seg = optim.Adam(model_seg.parameters())
    optim_gen = optim.Adam(model_gen.parameters())
    optim_discr = optim.Adam(model_discr.parameters())

    print('Initializing misc..')
    batch_count = 0
    partition_count = 0
    results = dict()
    early_stopper = EarlyStopper('accuracy', early_stop_patience)
    visualiser = vis.Visualiser(server, port, run_name, reload, visdom_dir)

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
                logits = model_seg(input.to(device))

                optimizer.zero_grad()
                loss_seg = obj_ce(
                    logits.permute(0, 2, 3, 1).contiguous().view(-1, num_classes),
                    labels.view(-1).to(device)
                )
                loss_seg.backward()
                optimizer.step()

                partition_loss += loss_seg.item()
                X, data = np.array([batch_count]), np.array([loss_seg.detach().to('cpu').numpy()])
                visualiser.plot(X, data, title='Loss per batch', legend=['Loss'], iteration=2, update='append')

            del logits
            del loss_seg
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            results['partition_avg_loss'] = np.divide(partition_loss, batch_per_part)
            results.update(evaluate(model_seg, sim_data, device))
            early_stopper.update(results, epoch_id, batch_count)
            log_and_viz_results(results, epoch_id, batch_count, partition_count, visualiser, start)

            if early_stopper.new_best and save_dir:
                model_seg.save(os.path.join(save_dir, '{}.pth'.format(seg_model.name)))

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

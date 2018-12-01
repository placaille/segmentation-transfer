import click
import sys
import torch
import os
import models
import itertools
import numpy as np

sys.path.append('src')

from torch import nn, optim
from data import PartitionProvider, InfiniteProviderFromPartitions
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
@click.option('--seg-model-path', type=click.Path(exists=True, dir_okay=False))
@click.option('--seg-model-name', default='segnet', type=click.Choice([
              'segnet', 'segnet_strided_upsample'
              ]), help='Name of model architecture (default segnet)')
@click.option('--save-dir', type=click.Path(writable=True, file_okay=False))
@click.option('--visdom-dir', type=click.Path(writable=True, file_okay=False))
@click.option('--config-file', type=click.Path(exists=True, dir_okay=False))
@click.option('--batch-size', type=int, default=32, help='(default 32)')
@click.option('--discr-model-name', default='dcgan_discr', help='Name of discriminator model (default dcgan_discr)',
            type=click.Choice(['dcgan_discr']))
@click.option('--gen-model-name', default='style_transfer_gen', help='Name of generator model (default style_transfer_gen)',
            type=click.Choice(['style_transfer_gen']))
@click.option('--early-stop-patience', type=int, default=10)
@click.option('--server', type=str, default=None)
@click.option('--port', type=str, default=None)
@click.option('--reload', is_flag=True,
               help='Flag notifying that the experiment is being reloaded.')
@click.option('--batch-per-eval', type=int, default=1500, help='default 1500')
def main(data_sim_dir, data_real_dir, data_label_dir, save_dir, visdom_dir, batch_size,
         config_file, discr_model_name, gen_model_name, early_stop_patience,
         server, port, reload, run_name, batch_per_eval, seg_model_path, seg_model_name):

    print('Loading data..')
    num_classes = 4
    input_channels = 3
    real_data = InfiniteProviderFromPartitions(
        input_dir=data_real_dir,
        label_dir=None,
        num_workers=0,
        partition_batch_size=batch_size,
        partition_num_workers=2
    )

    sim_data = InfiniteProviderFromPartitions(
        input_dir=data_sim_dir,
        label_dir=data_label_dir,
        num_workers=0,
        partition_batch_size=batch_size,
        partition_num_workers=2
    )

    real_data.init_iterator(train=True)
    sim_data.init_iterator(train=True)

    print('Building model & loading on GPU (if applicable)..')
    model_gen = models.get_generator_model(gen_model_name, input_channels).to(device)
    model_discr = models.get_discriminator_model(discr_model_name, input_channels).to(device)
    if save_dir:
        model_gen.save(os.path.join(save_dir, '{}.pth'.format(model_gen.name)))
        model_discr.save(os.path.join(save_dir, '{}.pth'.format(model_discr.name)))

    if seg_model_path:
        assert seg_model_name in seg_model_path
        model_seg = models.get_seg_model(seg_model_name, num_classes, input_channels).to(device)
        model_seg.load(seg_model_path)

    obj_adv = nn.BCELoss()
    label_true = 1
    label_fake = 0

    optim_gen = optim.Adam(model_gen.parameters())
    optim_discr = optim.Adam(model_discr.parameters())

    print('Initializing misc..')
    batch_count = 0
    results = dict()
    results['accuracy'] = 0
    early_stopper = EarlyStopper('accuracy', early_stop_patience)
    visualiser = vis.Visualiser(server, port, run_name, reload, visdom_dir)

    print('Starting training..')
    for eval_count in count(start=1):

        batch_since_eval = 0
        loss_discr_true = 0
        loss_discr_fake = 0
        loss_gen = 0
        model_gen.train()
        model_discr.train()
        start = timer()

        while batch_since_eval <= batch_per_eval:

            batch_real = next(real_data)
            batch_sim, _ = next(sim_data)
            assert batch_sim.shape[0] == batch_real.shape[0]
            b_size = batch_sim.shape[0]
            batch_count += 1

            loss_d = 0
            optim_discr.zero_grad()
            # train discriminator on true data (logD(x))
            scores_true = model_discr(batch_sim.to(device))
            labels = torch.full((b_size, ), label_true).to(device)
            loss_d_true = obj_adv(scores_true.view(-1), labels.to(device))
            loss_d_true.backward()
            loss_d += loss_d_true.item()
            loss_discr_true += loss_d_true.item()

            # train discriminator on fake data (log(1-D(G(x)))
            batch_fake = model_gen(batch_real.to(device))
            scores_fake = model_discr(batch_fake.detach())
            labels.fill_(label_fake)
            loss_d_fake = obj_adv(scores_fake.view(-1), labels)
            loss_d_fake.backward()
            loss_d += loss_d_fake.item()
            loss_discr_fake += loss_d_fake.item()

            optim_discr.step()

            loss_g = 0
            optim_gen.zero_grad()
            # train generator
            scores_fake = model_discr(batch_fake)
            labels.fill_(label_true)
            loss_g_fake = obj_adv(scores_fake.view(-1), labels)
            loss_g_fake.backward()
            loss_g += loss_g_fake.item()
            loss_gen += loss_g_fake.item()

            optim_gen.step()

            loss_tot = loss_d + loss_g
            X, data = np.array([batch_count]), np.array([loss_tot])
            visualiser.plot(X, data, title='Loss per batch', legend=['Loss'], iteration=2, update='append')

            batch_since_eval += 1

        # DO EVAL
        torch.cuda.empty_cache()
        eval_count += 1
        results['loss_gen'] = np.divide(loss_gen, batch_since_eval)
        results['loss_discr_fake'] = np.divide(loss_discr_fake, batch_since_eval)
        results['loss_discr_true'] = np.divide(loss_discr_true, batch_since_eval)

        if seg_model_path:
            results.update(evaluate(model_seg, model_gen, sim_data, device))
            early_stopper.update(results, epoch_id=eval_count, batch_id=batch_count)
        log_and_viz_results(results, batch_count, eval_count, visualiser, start)

        if early_stopper.new_best and save_dir:
            model_gen.save(os.path.join(save_dir, '{}.pth'.format(model_gen.name)))
            model_discr.save(os.path.join(save_dir, '{}.pth'.format(model_disc.name)))

        if early_stopper.stop:
            early_stopper.print_stop()
            return


def log_and_viz_results(results, batch_id, eval_count, visualiser, start):

    print('batch {:6d} - loss D(fake)/D(true)/G {:7.4f}/{:7.4f}/{:7.4f} - valid acc {:7.4f} - {:4.1f} secs' \
          .format(
              batch_id,
              results['loss_discr_fake'],
              results['loss_discr_true'],
              results['loss_gen'],
              results['accuracy'],
              timer()-start
          ))

    X = np.array([eval_count])
    data_d_fake = np.array([results['loss_discr_fake']])
    data_d_true = np.array([results['loss_discr_true']])
    data_gen = np.array([results['loss_gen']])

    visualiser.plot(X, data_d_fake, title='Discriminator fake loss', legend=['Loss'], iteration=0, update='append')
    visualiser.plot(X, data_d_true, title='Discriminator true loss', legend=['Loss'], iteration=1, update='append')
    visualiser.plot(X, data_gen, title='Generator loss', legend=['Loss'], iteration=3, update='append')

    data = np.array([results['accuracy']])
    visualiser.plot(X, data, title='Accuracy', legend=['Accuracy'], iteration=5, update='append')
    # visualiser.image(results['images'], title='Source image', iteration=0)
    # visualiser.image(results['targets'], title='Target image', iteration=1)
    # visualiser.image(results['segmented'], title='Segmented image', iteration=2)


if __name__ == '__main__':
    main()

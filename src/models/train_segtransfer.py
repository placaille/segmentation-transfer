import click
import sys
import torch
import os
import models
import numpy as np

sys.path.append('src')

from torch import nn, optim
from data import PartitionProvider, InfiniteProviderFromPartitions
from utils import CommandWithConfigFile, EarlyStopper
from utils import vis
from evaluate import evaluate_segtransfer
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
@click.option('--batch-per-eval', type=int, default=500, help='default 500')
@click.option('--batch-per-save', type=int, default=5000, help='default 5000')
@click.option('--max-num-batch', type=int, default=250000, help='default 250000')
@click.option('--content-weight', type=float, default=0.0001)
def main(data_sim_dir, data_real_dir, data_label_dir, save_dir, visdom_dir,
         batch_size, config_file, discr_model_name, gen_model_name,
         early_stop_patience, max_num_batch, server, port, reload, run_name,
         batch_per_eval, batch_per_save, seg_model_path, seg_model_name,
         content_weight):

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

    real_data.init_train_iterator()
    sim_data.init_train_iterator()

    print('Building model & loading on GPU (if applicable)..')
    if seg_model_path:
        assert seg_model_name in seg_model_path
        model_seg = models.get_seg_model(seg_model_name, num_classes, input_channels).to(device)
        model_seg.load(seg_model_path)

    model_discr = models.get_discriminator_model(discr_model_name, model_seg.size_bottleneck).to(device)
    if save_dir:
        model_discr.save(os.path.join(save_dir, '{}_{}.pth'.format(model_discr.name, 0)))

    obj_adv = nn.BCELoss()
    label_true = 1
    label_fake = 0

    optim_gen = optim.Adam(filter(lambda x: 'd' not in x.__class__.__name__, model_seg.parameters()))
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
        loss_discr_true_sum = 0
        loss_discr_fake_sum = 0
        loss_gen_sum = 0
        model_seg.train()
        model_discr.train()
        start = timer()

        while batch_since_eval < batch_per_eval:

            batch_real = next(real_data).to(device)
            batch_sim = next(sim_data)[0].to(device)
            assert batch_sim.shape[0] == batch_real.shape[0]
            b_size = batch_sim.shape[0]
            batch_count += 1

            optim_discr.zero_grad()
            # train discriminator on true data (logD(x))
            emb_sim = model_seg(batch_sim.to(device), bottleneck=True)
            print(batch_real.shape)
            #scores_true = model_discr(emb_sim.detach())
            #labels = torch.full((b_size, ), label_true).to(device)
            #loss_d_true = obj_adv(scores_true.view(-1), labels.to(device))
            #loss_d_true.backward()

            # train discriminator on fake data (log(1-D(G(x)))
            batch_fake = model_seg(batch_real.to(device), bottleneck=True)
            print(batch_fake.shape)
            exit(0)
            scores_fake = model_discr(batch_fake.detach())
            labels.fill_(label_fake)
            loss_d_fake = obj_adv(scores_fake.view(-1), labels)
            loss_d_fake.backward()

            optim_discr.step()

            optim_gen.zero_grad()
            # train generator
            scores_fake = model_discr(batch_fake)
            labels.fill_(label_true)
            loss_g_fake = obj_adv(scores_fake.view(-1), labels)
            loss_g_fake.backward()

            optim_gen.step()

            loss_discr_fake_sum += loss_d_fake.item()
            loss_discr_true_sum += loss_d_true.item()
            loss_gen_sum += loss_g_fake.item()
            loss_tot = loss_d_fake.item() + loss_d_true.item()

            X, data = np.array([batch_count]), np.array([loss_tot])
            visualiser.plot(X, data, title='Loss per batch', legend=['Loss'], iteration=2, update='append')

            batch_since_eval += 1

        # DO EVAL
        torch.cuda.empty_cache()
        eval_count += 1
        results['loss_gen'] = np.divide(loss_gen_sum, batch_since_eval)
        results['loss_discr_fake'] = np.divide(loss_discr_fake_sum, batch_since_eval)
        results['loss_discr_true'] = np.divide(loss_discr_true_sum, batch_since_eval)

        if seg_model_path:
            results.update(evaluate_segtransfer(model_seg, sim_data.get_valid_iterator(),
                                                real_data.get_valid_iterator(), device))
            # early_stopper.update(results, epoch_id=eval_count, batch_id=batch_count)
        log_and_viz_results(results, batch_count, eval_count, visualiser, start, save_dir)

        if save_dir and batch_count % batch_per_save == 0:
            # model_gen.save(os.path.join(save_dir, '{}_{}.pth'.format(model_gen.name, batch_count)))
            model_discr.save(os.path.join(save_dir, '{}_{}.pth'.format(model_discr.name, batch_count)))

        if batch_count >= max_num_batch:
            print('Stopping training..')
            break


def log_and_viz_results(results, batch_id, eval_count, visualiser, start, save_dir):

    print('batch {:6d} - loss D/G {:7.4f}/{:7.4f} - {:4.1f} secs' \
        .format(
        batch_id,
        results['loss_discr_fake'] + results['loss_discr_true'],
        results['loss_gen_style'] + results['loss_gen_content'],
        timer()-start
    ))

    X = np.array([eval_count])
    data_d_fake = np.array([results['loss_discr_fake']])
    data_d_true = np.array([results['loss_discr_true']])
    data_gen = np.array([results['loss_gen']])

    visualiser.plot(X, data_d_fake, title='Discriminator fake loss', legend=['Loss'], iteration=0, update='append')
    visualiser.plot(X, data_d_true, title='Discriminator true loss', legend=['Loss'], iteration=1, update='append')
    visualiser.plot(X, data_gen, title='Generator loss', legend=['Loss'], iteration=3, update='append')

    visualiser.image(results['real'], title='Source real image', iteration=0)
    visualiser.image(results['sim'], title='Source sim image', iteration=1)
    visualiser.image(results['real_segmented'], title='Segmented real image', iteration=2)
    visualiser.image(results['sim_segmented'], title='Segmented sim image', iteration=3)

    if save_dir:
        np.save(os.path.join(save_dir, 'img_source.npy'), results['real'])
        np.save(os.path.join(save_dir, 'img_transformed_{}.npy'.format(batch_id)), results['transformed'])
        np.save(os.path.join(save_dir, 'img_segmented_{}.npy'.format(batch_id)), results['segmented'])


if __name__ == '__main__':
    main()

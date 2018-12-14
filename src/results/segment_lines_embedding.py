import numpy as np
import imageio
import click
import torch
import os
import re
import sys
import cv2

sys.path.append('src')

from models import get_seg_model, get_generator_model, evaluate_transfer
from data import CustomDataset
from utils import logit_to_img

from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.nn.functional import interpolate
from timeit import default_timer as timer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def convert_batch_to_frames(source, segmented):
    frames = []
    for (src, sgt) in zip(source, segmented):
        stack = torch.stack((src, sgt))
        grid = make_grid(stack, nrow=2, padding=2, normalize=False)
        frames.append(grid.numpy().transpose(1, 2, 0))
    return frames


def get_frames(cap, frame_skip, max_frames):
    frames = []
    frame_count = 0
    while True:
        (grabbed, frame) = cap.read()
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if not grabbed:  # end of video
            break

        if frame_id % frame_skip == 0:
            frame_count += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(cv2.resize(frame_rgb, (160, 120)))

        if frame_id % 500 == 0:
            print('\tframes processed {}'.format(frame_id))
        if frame_id >= max_frames:
            break
    cap.release()
    return frames


@click.command()
@click.argument('input-file', type=click.Path(exists=True, dir_okay=False, readable=True))
@click.argument('output-file', type=click.Path(dir_okay=False, writable=True))
@click.argument('segmentation-model', type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option('--seg-model-name', default='segnet', type=str)
@click.option('--transform-model-name', default='style_transfer_gen', type=str)
@click.option('--frame-skip', type=int, default=3, help='Frames between captures')
@click.option('--batch-size', type=int, default=5)
@click.option('--max-frames', type=int, default=5500,
              help='Max number of frames processed (default 5500, approx. 3 minutes of video with frame skip = 3)')
def main(input_file, output_file, segmentation_model,
         seg_model_name, transform_model_name, frame_skip, batch_size,
         max_frames):
    """
    used to create a gif with lines segmented from a video files
    """
    assert output_file.endswith('.gif'), 'Make sure output_file is a .gif'

    print('Loading models..')
    num_classes = 4
    input_channels = 3
    model_seg = get_seg_model(seg_model_name, num_classes, input_channels).to(device)
    model_seg.load(segmentation_model)

    print('Loading data..')
    cap = cv2.VideoCapture(input_file)
    images = torch.tensor(get_frames(cap, frame_skip, max_frames)).to(torch.uint8)
    num_images = images.shape[0]
    data_iterator = DataLoader(
        dataset=CustomDataset(images),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )
    print('\tNumber of frames to convert:\t{} (frame skip: {})'.format(num_images, frame_skip))

    print('Converting..')
    model_seg.eval()
    batch_count = 0
    gif_frames = []
    with torch.no_grad():
        start = timer()
        for images in data_iterator:
            batch_count += 1

            # get segmentation
            seg_logits = model_seg(images.to(device))
            seg_preds = torch.argmax(seg_logits, dim=1).cpu()

            source = torch.mul(images.cpu(), 255).to(torch.uint8)
            segmented = logit_to_img(seg_preds.cpu().numpy()).transpose(0, 3, 1, 2)
            segmented = torch.mul(torch.tensor(segmented), 255).to(torch.uint8)

            # convert torch predictions to frames of grid
            gif_frames.extend(convert_batch_to_frames(source, segmented))

            if batch_count % 50 == 0:
                print('\tframe {} / {} - {:.2f} secs'.format(
                    batch_count*batch_size, num_images, timer() - start)
                )
                start = timer()

        del images, source, segmented,
        # convert sequence of frames into gif
        print('Saving {}..'.format(output_file))
        imageio.mimsave(output_file, gif_frames, fps=29.97/frame_skip, subrectangles=True)


if __name__ == '__main__':
    main()

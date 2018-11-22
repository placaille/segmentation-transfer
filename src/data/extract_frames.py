import cv2
import click
import os
import numpy as np
import h5py


def get_frames(cap, frame_skip):
    frames = []
    nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_id in range(0, nb_frames, frame_skip):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        frames.append(cv2.resize(frame, (160, 120)))
    cap.release()
    return frames


@click.command()
@click.argument('video-dir', type=click.Path(exists=True, file_okay=False))
@click.argument('output-file', type=click.Path(writable=True, dir_okay=False))
@click.option('--frame-skip', type=int, default=25, help='Frames to skip between captures', )
def main(video_dir, output_file, frame_skip):
    frames = []
    for file in os.listdir(video_dir):
        if file.endswith('.mp4'):
            print('\tProcessing {}..'.format(file))
            cap = cv2.VideoCapture(os.path.join(video_dir, file))
            frames.extend(get_frames(cap, frame_skip))

    stack = np.stack(frames)
    print(stack.shape)
    f = h5py.File(output_file, 'w')
    dset = f.create_dataset('raw_real', (stack.shape[0],120,160,3), dtype='int16', data=stack)

if __name__ == '__main__':
    main()

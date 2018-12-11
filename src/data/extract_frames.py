import cv2
import click
import os
import numpy as np


def get_frames(cap, frame_skip):
    frames = []
    nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_id in range(0, nb_frames, frame_skip):
        if frame_id % 1000 == 0:
            print('\t\t frame {}'.format(frame_id))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(frame_rgb, (160, 120)))
    cap.release()
    return frames


@click.command()
@click.argument('video-dir', type=click.Path(exists=True, file_okay=False))
@click.argument('output-file', type=click.Path(writable=True, dir_okay=False))
@click.option('--frame-skip', type=int, default=25, help='Frames to skip between captures', )
def main(video_dir, output_file, frame_skip):
    frames = []
    used_videos_file = open(os.path.join(video_dir, 'used_videos.info'))
    with open(used_videos_file, 'r') as file:
        used_videos = file.read().splitlines()

    for file in os.listdir(video_dir):
        if file.endswith('.mp4') and file not in used_videos:
            print('\tProcessing {}..'.format(file))
            cap = cv2.VideoCapture(os.path.join(video_dir, file))
            frames.extend(get_frames(cap, frame_skip))
            with open(used_videos_file, 'a') as list_file:
                list_file.write(file + '\n')

    stack = np.stack(frames).astype(np.int16)
    print(stack.shape)
    np.save(output_file, stack)

if __name__ == '__main__':
    main()

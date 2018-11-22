import requests
import os
import click


@click.command()
@click.argument('url')
@click.argument('list-of-videos', type=click.Path(exists=True, dir_okay=False))
@click.argument('save-dir', type=click.Path(exists=True, file_okay=False, writable=True))
def main(url, list_of_videos, save_dir):
    try:
        with open(list_of_videos, 'r') as file:
            video_names = file.read().splitlines()

        print('Downloading videos from {}'.format(url))
        for video_name in video_names:
            file_name = video_name + '.video.mp4'
            response = requests.get(url + file_name)
            with open(os.path.join(save_dir, file_name), 'wb') as file:
                file.write(response.content)
            print('\t{} done.'.format(file_name))
    except:
        pass
    else:
        with open(os.path.join(save_dir, 'download.info'), 'w') as file:
            file.write('Success')


if __name__ == '__main__':
    main()

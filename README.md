# segmentation-transfer

Data files `classes.npy`, `raw.npy` and `segmented.npy` must be placed in `data/raw`

# Real Duckiebot images

The database of logs can be found [here](http://ipfs.duckietown.org:8080/ipfs/QmUbtwQ3QZKmmz5qTjKM3z8LJjsrKBWLUnnzoE5L4M7y7J/logs/gallery.html). Files can be directly downloaded from [here](https://gateway.ipfs.io/ipfs/QmUbtwQ3QZKmmz5qTjKM3z8LJjsrKBWLUnnzoE5L4M7y7J/logs/) using the following command.

```
make data/videos/download.info
```

A list of videos used is listed in the file [`data/videos/list_of_videos.txt`](data/videos/list_of_videos.txt).

## Extracting frames

Frames were extracted from the raw videos from the logs and downsampled from 640x480 to 160x120. Extracting every 10 frames of the downloaded videos provided a dataset of XXX images.

To extract the frames from the set of downloaded videos, simply use the following command.

```
make data/raw/raw_real.hdf5
```

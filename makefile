.PHONY: all clean clear-models segnet


all: models/basic_encoder_decoder.pth

clean: clear-models
	rm -f data/processed/*.pkl
	rm -f data/raw/data_info.txt

clear-models:
	rm -f models/*.pth

data/raw/data_info.txt:
	python src/data/explore.py data/raw/raw.npy data/raw/data_info.txt
	python src/data/explore.py data/raw/segmented.npy data/raw/data_info.txt
	python src/data/explore.py data/raw/classes.npy data/raw/data_info.txt

# tuple of (train, valid, test) TensorDataset
data/processed/processed.pkl:  data/raw/data_info.txt
	python src/data/preprocess.py \
		data/raw/raw.npy data/raw/classes.npy \
		data/processed/processed.pkl

data/videos/download.info:
	python src/data/download_videos.py \
	https://gateway.ipfs.io/ipfs/QmUbtwQ3QZKmmz5qTjKM3z8LJjsrKBWLUnnzoE5L4M7y7J/logs/ \
	data/videos/list_of_videos.txt \
	data/videos

data/raw/raw_real.hdf5: data/videos/download.info
	python src/data/extract_frames.py data/videos data/raw/raw_real.hdf5 \
	--frame-skip 10

# train and save
models/segnet.pth: data/processed/processed.pkl
	python src/models/train.py data/processed/processed.pkl \
	--save-dir=models \
	--model-name=segnet

# train and save
segnet: data/processed/processed.pkl
	python src/models/train.py data/processed/processed.pkl \
	--model-name=segnet

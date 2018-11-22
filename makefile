.PHONY: all clean local test clear-models segnet

# default mila directories
DATA_DIR=/data/lisa/data/duckietown-segmentation/data
MODEL_DIR=/data/milatmp1/$(USER)/duckietown-segmentation/models

# to run in local repository
local:
	$(eval DATA_DIR=./data)
	$(eval MODEL_DIR=./models)
	@echo Setting DATA_DIR and MODEL_DIR to ./data and ./models

all: $(MODEL_DIR)/segnet.pth

clean: clear-models
	rm -f $(DATA_DIR)/processed/*.pkl
	rm -f $(DATA_DIR)/raw/data_info.txt

clear-models:
	rm -f $(MODEL_DIR)/*.pth

data/raw/data_info.txt:
	python src/data/explore.py $(DATA_DIR)/raw/raw.npy $(DATA_DIR)/raw/data_info.txt
	python src/data/explore.py $(DATA_DIR)/raw/segmented.npy $(DATA_DIR)/raw/data_info.txt
	python src/data/explore.py $(DATA_DIR)/raw/classes.npy $(DATA_DIR)/raw/data_info.txt

# tuple of (train, valid, test) TensorDataset
data/processed/processed.pkl:  $(DATA_DIR)/raw/data_info.txt
	python src/data/preprocess.py \
		$(DATA_DIR)/raw/raw.npy $(DATA_DIR)/raw/classes.npy \
		$(DATA_DIR)/processed/processed.pkl

data/videos/download.info:
	python src/data/download_videos.py \
	https://gateway.ipfs.io/ipfs/QmUbtwQ3QZKmmz5qTjKM3z8LJjsrKBWLUnnzoE5L4M7y7J/logs/ \
	data/videos/list_of_videos.txt \
	$(DATA_DIR)/videos

data/raw/raw_real.hdf5: $(DATA_DIR)/videos/download.info
	python src/data/extract_frames.py $(DATA_DIR)/videos $(DATA_DIR)/raw/raw_real.hdf5 \
	--frame-skip 10

# train and save
models/segnet.pth: $(DATA_DIR)/processed/processed.pkl
	python src/models/train.py $(DATA_DIR)/processed/processed.pkl \
	--save-dir=$(MODEL_DIR) \
	--model-name=segnet

# train only (no save)
segnet: $(DATA_DIR)/processed/processed.pkl
	python src/models/train.py $(DATA_DIR)/processed/processed.pkl \
	--model-name=segnet

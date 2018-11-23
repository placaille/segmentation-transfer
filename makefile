.PHONY: all clean local clear-models segnet

# to run in local repository
local=false
ifeq ($(local), false)
DATA_DIR=/data/lisa/data/duckietown-segmentation/data
MODEL_DIR=/data/milatmp1/$(USER)/duckietown-segmentation/models
else
DATA_DIR=./data
MODEL_DIR=./models
endif

# hack for pointing to files
data/processed/processed.pkl=$(DATA_DIR)/processed/processed.pkl
data/raw/data_info.txt=$(DATA_DIR)/raw/data_info.txt
data/videos/download.info=$(DATA_DIR)/videos/download.info
data/raw/raw_real.hdf5=$(DATA_DIR)/raw/raw_real.hdf5
models/segnet.pth=$(MODEL_DIR)/segnet.pth


all: data/raw/data_info.txt

clean: clear-models
	rm -f $(DATA_DIR)/processed/*.pkl
	rm -f $(DATA_DIR)/raw/data_info.txt

clear-models:
	rm -f $(MODEL_DIR)/*.pth

data/raw/data_info.txt: $(data/raw/data_info.txt)
$(data/raw/data_info.txt):
	python src/data/explore.py $(DATA_DIR)/raw/raw_real.hdf5 $(DATA_DIR)/raw/data_info.txt
	python src/data/explore.py $(DATA_DIR)/raw/raw_sim.npy $(DATA_DIR)/raw/data_info.txt
	python src/data/explore.py $(DATA_DIR)/raw/classes.npy $(DATA_DIR)/raw/data_info.txt

# pre-process data to TensorDataset
data/processed/processed.pkl:$(data/processed/processed.pkl)
$(data/processed/processed.pkl): $(data/raw/data_info.txt)
	python src/data/preprocess.py \
	$(DATA_DIR)/raw/raw_real.hdf5 $(DATA_DIR)/raw/raw_sim.npy $(DATA_DIR)/raw/classes.npy \
	$(DATA_DIR)/processed/processed.pkl

# download videos
data/videos/download.info:$(data/videos/download.info)
$(data/videos/download.info):
	python src/data/download_videos.py \
	https://gateway.ipfs.io/ipfs/QmUbtwQ3QZKmmz5qTjKM3z8LJjsrKBWLUnnzoE5L4M7y7J/logs/ \
	data/videos/list_of_videos.txt \
	$(DATA_DIR)/videos

# extract frames
data/raw/raw_real.hdf5:$(data/raw/raw_real.hdf5)
$(data/raw/raw_real.hdf5): $(data/videos/download.info)
	python src/data/extract_frames.py $(DATA_DIR)/videos $(DATA_DIR)/raw/raw_real.hdf5 \
	--frame-skip 10

# train and save
models/segnet.pth:$(models/segnet.pth)
$(models/segnet.pth): $(data/processed/processed.pkl)
	python src/models/train.py $(DATA_DIR)/processed/processed.pkl \
	--save-dir=$(MODEL_DIR) \
	--model-name=segnet

# train only (no save)
segnet: processed/processed.pkl
	python src/models/train.py $(DATA_DIR)/processed/processed.pkl \
	--model-name=segnet

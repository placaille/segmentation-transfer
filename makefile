.PHONY: all clean local segnet

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
data/videos/download.info=$(DATA_DIR)/videos/download.info
data/hdf5/real.hdf5=$(DATA_DIR)/hdf5/real.hdf5
data/videos/real.hdf5=$(DATA_DIR)/videos/real.hdf5
models/segnet.pth=$(MODEL_DIR)/segnet.pth


all: models/segnet.pth

clean:
	rm -f $(MODEL_DIR)/*.pth
	rm -f $(DATA_DIR)/hdf5/data_info.txt

# download videos
data/videos/download.info:$(data/videos/download.info)
$(data/videos/download.info):
	python src/data/download_videos.py \
	https://gateway.ipfs.io/ipfs/QmUbtwQ3QZKmmz5qTjKM3z8LJjsrKBWLUnnzoE5L4M7y7J/logs/ \
	data/videos/list_of_videos.txt \
	$(DATA_DIR)/videos

# extract frames
data/videos/real.hdf5:$(data/videos/real.hdf5)
$(data/videos/real.hdf5): $(data/videos/download.info)
	python src/data/extract_frames.py $(DATA_DIR)/videos $(DATA_DIR)/hdf5/real.hdf5 \
	--frame-skip 10

# split train/valid (ONLY REMOTE)
data/hdf5/real.hdf5:$(data/hdf5/real.hdf5)
$(data/hdf5/real.hdf5): $(data/videos/real.hdf5)
	python src/data/split.py

# train and save
models/segnet.pth:$(models/segnet.pth)
$(models/segnet.pth): $(data/hdf5/real.hdf5)
	python src/models/train.py \
	$(DATA_DIR)/hdf5/sim.hdf5 $(DATA_DIR)/hdf5/real.hdf5 $(DATA_DIR)/hdf5/classes.hdf5 \
	--save-dir=$(MODEL_DIR) \
	--seg-model-name=segnet

# train only (no save)
segnet: $(data/hdf5/real.hdf5)
	python src/models/train.py \
	$(DATA_DIR)/hdf5/sim.hdf5 $(DATA_DIR)/hdf5/real.hdf5 $(DATA_DIR)/hdf5/classes.hdf5 \
	--seg-model-name=segnet

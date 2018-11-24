.PHONY: all clean segnet tiny-segnet

# to run in local repository
local=false
ifeq ($(local), false)
DATA_DIR=/data/lisa/data/duckietown-segmentation/data
MODEL_DIR=/data/milatmp1/$(USER)/duckietown-segmentation/models
TMP_DATA_DIR=/Tmp/$(USER)/segmentation-transfer/data
else
DATA_DIR=./data
MODEL_DIR=./models
TMP_DATA_DIR=/tmp/segmentation-transfer/data
endif

# hack for pointing to files
data/videos/download.info=$(DATA_DIR)/videos/download.info
data/videos/real.hdf5=$(DATA_DIR)/videos/real.hdf5
data/hdf5/real.hdf5=$(DATA_DIR)/hdf5/real.hdf5
data/hdf5/classes.hdf5=$(DATA_DIR)/hdf5/classes.hdf5
data/hdf5/sim.hdf5=$(DATA_DIR)/hdf5/sim.hdf5
data/hdf5/.sentinel=$(DATA_DIR)/hdf5/.sentinel
data/hdf5_tiny/.sentinel=$(DATA_DIR)/hdf5_tiny/.sentinel
tmp/data/hdf5/.sentinel=$(TMP_DATA_DIR)/hdf5/.sentinel
tmp/data/hdf5_tiny/.sentinel=$(TMP_DATA_DIR)/hdf5_tiny/.sentinel
models/segnet.pth=$(MODEL_DIR)/segnet.pth


all: models/segnet.pth

clean:
	rm -f $(MODEL_DIR)/*.pth

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

# split real train/valid (ONLY REMOTE)
data/hdf5/real.hdf5:$(data/hdf5/real.hdf5)
$(data/hdf5/real.hdf5): $(data/videos/real.hdf5)
	python src/data/split.py

# make sentinel if all data is present
data/hdf5/.sentinel:$(data/hdf5/.sentinel)
$(data/hdf5/.sentinel): $(data/hdf5/real.hdf5) $(data/hdf5/classes.hdf5) $(data/hdf5/sim.hdf5)
	@touch $@

# make sentinel if all data is present
data/hdf5_tiny/.sentinel:$(data/hdf5_tiny/.sentinel)
$(data/hdf5_tiny/.sentinel): $(data/hdf5_tiny/real_tiny.hdf5) $(data/hdf5_tiny/classes_tiny.hdf5) $(data/hdf5_tiny/sim_tiny.hdf5)
	@touch $@

# copy to tmp location
tmp/data/hdf5/.sentinel:$(tmp/data/hdf5/.sentinel)
$(tmp/data/hdf5/.sentinel): $(data/hdf5/.sentinel)
	mkdir -p $(TMP_DATA_DIR)/hdf5
	cp $(DATA_DIR)/hdf5/sim.hdf5 $(TMP_DATA_DIR)/hdf5/
	cp $(DATA_DIR)/hdf5/real.hdf5 $(TMP_DATA_DIR)/hdf5/
	cp $(DATA_DIR)/hdf5/classes.hdf5 $(TMP_DATA_DIR)/hdf5/
	@touch $@

# copy tiny to tmp location
tmp/data/hdf5_tiny/.sentinel:$(tmp/data/hdf5_tiny/.sentinel)
$(tmp/data/hdf5_tiny/.sentinel): $(data/hdf5_tiny/.sentinel)
	mkdir -p $(TMP_DATA_DIR)/hdf5_tiny
	cp $(DATA_DIR)/hdf5_tiny/sim_tiny.hdf5 $(TMP_DATA_DIR)/hdf5_tiny/
	cp $(DATA_DIR)/hdf5_tiny/real_tiny.hdf5 $(TMP_DATA_DIR)/hdf5_tiny/
	cp $(DATA_DIR)/hdf5_tiny/classes_tiny.hdf5 $(TMP_DATA_DIR)/hdf5_tiny/
	@touch $@

# train and save
models/segnet.pth:$(models/segnet.pth)
$(models/segnet.pth): $(tmp/data/hdf5/.sentinel)
	python src/models/train.py \
	$(TMP_DATA_DIR)/hdf5/sim.hdf5 \
	$(TMP_DATA_DIR)/hdf5/real.hdf5 \
	$(TMP_DATA_DIR)/hdf5/classes.hdf5 \
	--save-dir=$(MODEL_DIR) \
	--seg-model-name=segnet

# train only (no save)
segnet: $(tmp/data/hdf5/.sentinel)
	python src/models/train.py \
	$(TMP_DATA_DIR)/hdf5/sim.hdf5 \
	$(TMP_DATA_DIR)/hdf5/real.hdf5 \
	$(TMP_DATA_DIR)/hdf5/classes.hdf5 \
	--seg-model-name=segnet

# train only (no save)
tiny-segnet: $(tmp/data/hdf5_tiny/.sentinel)
	python src/models/train.py \
	$(TMP_DATA_DIR)/hdf5_tiny/sim_tiny.hdf5 \
	$(TMP_DATA_DIR)/hdf5_tiny/real_tiny.hdf5 \
	$(TMP_DATA_DIR)/hdf5_tiny/classes_tiny.hdf5 \
	--seg-model-name=segnet

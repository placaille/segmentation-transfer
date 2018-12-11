.PHONY: all clean segnet tiny-segnet segnet_strided_upsample test

# default args
local=false
run_name=default
config_file=template_config.yml

# to run in local repository
ifeq ($(local), false)
DATA_DIR=/data/lisa/data/duckietown-segmentation/data
MODEL_DIR=/data/milatmp1/$(USER)/duckietown-segmentation/models/$(run_name)
VISDOM_DIR=/data/milatmp1/$(USER)/duckietown-segmentation/visdom/$(run_name)
PRE_TRAINED_PATH=/data/milatmp1/$(USER)/duckietown-segmentation/models/pre-trained
TMP_DATA_DIR=/Tmp/$(USER)/segmentation-transfer/data
else
DATA_DIR=./data
MODEL_DIR=./models/$(run_name)
VISDOM_DIR=./visdom/$(run_name)
PRE_TRAINED_PATH=./models/pre-trained
TMP_DATA_DIR=/tmp/segmentation-transfer/data
endif

# hack for pointing to files
data/videos/download.info=$(DATA_DIR)/videos/download.info
data/videos/real.hdf5=$(DATA_DIR)/videos/real.hdf5
data/hdf5/real.hdf5=$(DATA_DIR)/hdf5/real.hdf5
data/split/real/.sentinel=$(DATA_DIR)/split/real/.sentinel
data/split/class/.sentinel=$(DATA_DIR)/split/class/.sentinel
data/split/sim/.sentinel=$(DATA_DIR)/split/sim/.sentinel
data/split/.sentinel=$(DATA_DIR)/split/.sentinel
data/split_tiny/.sentinel=$(DATA_DIR)/split_tiny/.sentinel
tmp/data/split/.sentinel=$(TMP_DATA_DIR)/split/.sentinel
tmp/data/split_tiny/.sentinel=$(TMP_DATA_DIR)/split_tiny/.sentinel
models/segnet.pth=$(MODEL_DIR)/segnet.pth
models/segnet_strided_upsample.pth=$(MODEL_DIR)/segnet_strided_upsample.pth


all: models/segnet.pth

clean:
	rm -f $(MODEL_DIR)/*.pth
	rm -f $(VISDOM_DIR)/*.out

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
data/split/.sentinel:$(data/split/.sentinel)
$(data/split/.sentinel): $(data/split/real/.sentinel) $(data/split/class/.sentinel) $(data/split/sim/.sentinel)
	@touch $@

# make sentinel if all data is present
data/split_tiny/.sentinel:$(data/split_tiny/.sentinel)
$(data/split_tiny/.sentinel): $(data/split_tiny/real/.sentinel) $(data/split_tiny/class/.sentinel) $(data/split_tiny/sim/.sentinel)
	@touch $@

# copy to tmp location
tmp/data/split/.sentinel:$(tmp/data/split/.sentinel)
$(tmp/data/split/.sentinel): $(data/split/.sentinel)
	mkdir -p $(TMP_DATA_DIR)/split
	cp -r $(DATA_DIR)/split $(TMP_DATA_DIR)
	@touch $@

# copy tiny to tmp location
tmp/data/split_tiny/.sentinel:$(tmp/data/split_tiny/.sentinel)
$(tmp/data/split_tiny/.sentinel): $(data/split_tiny/.sentinel)
	mkdir -p $(TMP_DATA_DIR)/split_tiny
	cp -r $(DATA_DIR)/split_tiny $(TMP_DATA_DIR)
	@touch $@

# train and save
models/segnet.pth:$(models/segnet.pth)
$(models/segnet.pth): $(tmp/data/split/.sentinel)
	mkdir -p $(MODEL_DIR)
	mkdir -p $(VISDOM_DIR)
	python src/models/train_segmentation.py \
	$(TMP_DATA_DIR)/split/sim \
	$(TMP_DATA_DIR)/split/real \
	$(TMP_DATA_DIR)/split/class \
	--run-name=$(run_name) \
	--save-dir=$(MODEL_DIR) \
	--visdom-dir=$(VISDOM_DIR) \
	--config-file=$(config_file) \
	--seg-model-name=segnet

# train only (no save)
segnet: $(tmp/data/split/.sentinel)
	python src/models/train_segmentation.py \
	$(TMP_DATA_DIR)/split/sim \
	$(TMP_DATA_DIR)/split/real \
	$(TMP_DATA_DIR)/split/class \
	--run-name=$(run_name) \
	--config-file=$(config_file) \
	--seg-model-name=segnet

# train and save
models/segnet_strided_upsample.pth:$(models/segnet_strided_upsample.pth)
$(models/segnet_strided_upsample.pth): $(tmp/data/split/.sentinel)
	mkdir -p $(MODEL_DIR)
	mkdir -p $(VISDOM_DIR)
	python src/models/train_segmentation.py \
	$(TMP_DATA_DIR)/split/sim \
	$(TMP_DATA_DIR)/split/real \
	$(TMP_DATA_DIR)/split/class \
	--run-name=$(run_name) \
	--save-dir=$(MODEL_DIR) \
	--visdom-dir=$(VISDOM_DIR) \
	--config-file=$(config_file) \
	--seg-model-name=segnet_strided_upsample

# train only (no save)
segnet_strided_upsample: $(tmp/data/split/.sentinel)
	python src/models/train_segmentation.py \
	$(TMP_DATA_DIR)/split/sim \
	$(TMP_DATA_DIR)/split/real \
	$(TMP_DATA_DIR)/split/class \
	--run-name=$(run_name) \
	--config-file=$(config_file) \
	--seg-model-name=segnet_strided_upsample

# train only (no save)
tiny-segnet: $(tmp/data/split_tiny/.sentinel)
	python src/models/train.py \
	$(TMP_DATA_DIR)/split_tiny/sim \
	$(TMP_DATA_DIR)/split_tiny/real \
	$(TMP_DATA_DIR)/split_tiny/class \
	--run-name=$(run_name) \
	--config-file=$(config_file) \
	--seg-model-name=segnet

# train only (no save)
tiny-transfer: $(tmp/data/split_tiny/.sentinel)
	python src/models/train_transfer.py \
	$(TMP_DATA_DIR)/split_tiny/sim \
	$(TMP_DATA_DIR)/split_tiny/real \
	$(TMP_DATA_DIR)/split_tiny/class \
	--run-name=$(run_name) \
	--config-file=$(config_file) \
	--seg-model-path=$(PRE_TRAINED_PATH)/segnet.pth \
	--batch-per-eval=1

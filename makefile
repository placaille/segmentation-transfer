.PHONY: segnet tiny-segnet segnet_strided_upsample tiny-transfer transfer models/style_transfer_gen.pth gif-transformed gif-embedding

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
data/videos/real.npy=$(DATA_DIR)/videos/real.npy
data/split/real/.sentinel=$(DATA_DIR)/split/real/.sentinel
data/split/class/.sentinel=$(DATA_DIR)/split/class/.sentinel
data/split/sim/.sentinel=$(DATA_DIR)/split/sim/.sentinel
data/split/.sentinel=$(DATA_DIR)/split/.sentinel
data/split_tiny/.sentinel=$(DATA_DIR)/split_tiny/.sentinel
tmp/data/split/.sentinel=$(TMP_DATA_DIR)/split/.sentinel
tmp/data/split_tiny/.sentinel=$(TMP_DATA_DIR)/split_tiny/.sentinel
models/segnet.pth=$(MODEL_DIR)/segnet.pth
models/segnet_strided_upsample.pth=$(MODEL_DIR)/segnet_strided_upsample.pth


# download videos
data/videos/download.info:$(data/videos/download.info)
$(data/videos/download.info):
	python src/data/download_videos.py \
	https://gateway.ipfs.io/ipfs/QmUbtwQ3QZKmmz5qTjKM3z8LJjsrKBWLUnnzoE5L4M7y7J/logs/ \
	data/videos/list_of_videos.txt \
	$(DATA_DIR)/videos

# extract frames
data/videos/real.npy:$(data/videos/real.npy)
$(data/videos/real.npy): $(data/videos/download.info)
	python src/data/extract_frames.py $(DATA_DIR)/videos $(DATA_DIR)/videos/real.npy \
	--frame-skip 10

# split real train/valid
data/split/real/.sentinel:$(data/split/real/.sentinel)
$(data/split/real/.sentinel): $(data/videos/real.npy)
	python src/data/split.py --data-dir $(DATA_DIR)
	@touch $@

# make sentinel if all data is present
data/split/.sentinel:$(data/split/.sentinel)
$(data/split/.sentinel): $(data/split/real/.sentinel)
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
models/style_transfer_gen.pth: $(tmp/data/split/.sentinel)
	mkdir -p $(MODEL_DIR)
	mkdir -p $(VISDOM_DIR)
	python src/models/train_transfer.py \
	$(TMP_DATA_DIR)/split/sim \
	$(TMP_DATA_DIR)/split/real \
	$(TMP_DATA_DIR)/split/class \
	--run-name=$(run_name) \
	--save-dir=$(MODEL_DIR) \
	--visdom-dir=$(VISDOM_DIR) \
	--config-file=$(config_file) \
	--seg-model-name=segnet \
	--seg-model-path=$(PRE_TRAINED_PATH)/segnet.pth \
	--discr-model-name=dcgan_discr \
	--gen-model-name=style_transfer_gen

# train only (no save)
transfer: $(tmp/data/split/.sentinel)
	python src/models/train_transfer.py \
	$(TMP_DATA_DIR)/split/sim \
	$(TMP_DATA_DIR)/split/real \
	$(TMP_DATA_DIR)/split/class \
	--run-name=$(run_name) \
	--config-file=$(config_file) \
	--seg-model-path=$(PRE_TRAINED_PATH)/segnet.pth

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

# download models (only local)
models/pre-trained/.sentinel:
	@echo Downloading pre-trained models..
	wget https://bit.ly/2BlpeBw -O ./models/pre-trained/segnet.pth -q --show-progress
	wget https://bit.ly/2QwzzVR -O ./models/pre-trained/dcgan_discr.pth -q --show-progress
	wget https://bit.ly/2EsPvSH -O ./models/pre-trained/style_transfer_gen.pth -q --show-progress
	#wget ............ embedding-model
	@touch $@

ifndef output_file
output_file=./results/segmented_lines.gif
endif
# convert input file to segmented
gif-transformed: models/pre-trained/.sentinel
	@echo Making $(input_file) into $(output_file) ..
	python src/results/segment_lines.py \
	$(input_file) \
	$(output_file) \
	./models/pre-trained/segnet.pth \
	./models/pre-trained/style_transfer_gen.pth

gif-embedding: models/pre-trained/.sentinel
	#TODO

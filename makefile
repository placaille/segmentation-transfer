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

# train and save
models/segnet.pth: data/processed/processed.pkl
	python src/models/train.py data/processed/processed.pkl \
	--save-dir=models \
	--model-name=segnet

# train and save
segnet: data/processed/processed.pkl
	python src/models/train.py data/processed/processed.pkl \
	--model-name=segnet

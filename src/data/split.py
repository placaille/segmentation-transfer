import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir')
args = parser.parse_args()

src_file = os.path.join(args.data_dir, 'videos/real.npy')
tgt_train_file = os.path.join(args.data_dir, 'split/real/train/real.train.0.npy')
tgt_valid_file = os.path.join(args.data_dir, 'split/real/valid/real.valid.0.npy')

all_data = np.load(src_file)

N = all_data.shape[0]
Ntrain = int(N * 0.9)

all_ids = np.arange(N)
np.random.shuffle(all_ids)

train_ids = all_ids[:Ntrain]
valid_ids = all_ids[Ntrain:]

np.save(tgt_train_file, all_data[train_ids])
np.save(tgt_valid_file, all_data[valid_ids])

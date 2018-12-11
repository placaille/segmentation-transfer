import numpy as np

src_file = '/data/lisa/data/duckietown-segmentation/data/videos/real.npy'
tgt_train_file = '/data/lisa/data/duckietown-segmentation/data/split/real/train/real.train.0.npy'
tgt_valid_file = '/data/lisa/data/duckietown-segmentation/data/split/real/valid/real.valid.0.npy'

all_data = np.load(src_file)

N = all_data.shape[0]
Ntrain = int(N * 0.9)

all_ids = np.arange(N)
np.random.shuffle(all_ids)

train_ids = all_ids[:Ntrain]
valid_ids = all_ids[Ntrain:]

np.save(tgt_train_file, all_data[train_ids])
np.save(tgt_valid_file, all_data[valid_ids])

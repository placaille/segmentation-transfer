import h5py
import numpy as np


f1 = h5py.File('/data/lisa/data/duckietown-segmentation/data/videos/real.hdf5')
f2 = h5py.File('/data/lisa/data/duckietown-segmentation/data/hdf5/real.hdf5', 'w')

N = 11419
Ntrain = int(N * 0.9)
Nvalid = N - Ntrain

all = np.arange(N)
np.random.shuffle(all)

train_ids = all[:Ntrain]
valid_ids = all[Ntrain:]

d_train = f2.create_dataset('train', (Ntrain, 120, 160, 3), dtype='int16')
d_valid = f2.create_dataset('valid', (Nvalid, 120, 160, 3), dtype='int16')

for pos, idx in enumerate(train_ids):
    img = f1['raw_real'][idx]
    d_train[pos] = img

for pos, idx in enumerate(valid_ids):
    img = f1['raw_real'][idx]
    d_valid[pos] = img

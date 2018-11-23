import h5py


f1 = h5py.File('data/videos/real.hdf5')
f2 = h5py.File('data/hdf5/real.hdf5', 'w')

import pdb;pdb.set_trace()
N = 11419
Ntrain = int(N * 0.9)
Nvalid = N - Ntrain

d_train = f2.create_dataset('train', (Ntrain, 120, 160,3), dtype='int16')
d_valid = f2.create_dataset('valid', (Nvalid, 120, 160,3), dtype='int16')

for i in range(Ntrain):
    if i % 1000 == 0:
        print(i)
    img = f1['raw_real'][i]
    d_train[i] = img

for i in range(Ntrain, N):
    if i % 1000 == 0:
        print(i)
    img = f1['raw_real'][i]
    d_valid[i - Ntrain] = img

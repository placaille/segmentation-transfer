import h5py
import os

dir = '/data/lisa/data/duckietown-segmentation/data/hdf5'
f_classes = h5py.File(os.path.join(dir, 'classes.hdf5'))
f_sim = h5py.File(os.path.join(dir, 'sim.hdf5'))
f_real = h5py.File(os.path.join(dir, 'real.hdf5'))

f_classes_tiny = h5py.File(os.path.join(dir, 'classes_tiny.hdf5'))
f_sim_tiny = h5py.File(os.path.join(dir, 'sim_tiny.hdf5'))
f_real_tiny = h5py.File(os.path.join(dir, 'real_tiny.hdf5'))


N = 128

dset = f_classes_tiny.create_dataset('train', data=f_classes['train'][:N])
dset = f_classes_tiny.create_dataset('valid', data=f_classes['valid'][:N])

dset = f_sim_tiny.create_dataset('train', data=f_sim['train'][:N])
dset = f_sim_tiny.create_dataset('valid', data=f_sim['valid'][:N])

dset = f_real_tiny.create_dataset('train', data=f_real['train'][:N])
dset = f_real_tiny.create_dataset('valid', data=f_real['valid'][:N])

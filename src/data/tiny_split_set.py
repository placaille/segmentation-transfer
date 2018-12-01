import numpy as np
import os

src_dir = '/data/lisa/data/duckietown-segmentation/data/split'
tgt_dir = '/data/lisa/data/duckietown-segmentation/data/split_tiny'
train_ext = 'train.0.npy'
valid_ext = 'valid.0.npy'

src_class_train = os.path.join(src_dir, 'class/train', 'classes.' + train_ext)
src_class_valid = os.path.join(src_dir, 'class/valid', 'classes.' + valid_ext)
src_sim_train = os.path.join(src_dir, 'sim/train', 'sim.' + train_ext)
src_sim_valid = os.path.join(src_dir, 'sim/valid', 'sim.' + valid_ext)
src_real_train = os.path.join(src_dir, 'real/train', 'real.' + train_ext)
src_real_valid = os.path.join(src_dir, 'real/valid', 'real.' + valid_ext)

src = [src_class_train, src_class_valid, src_sim_train, src_sim_valid, src_real_train, src_real_valid]

tgt_class_train = os.path.join(tgt_dir, 'class/train', 'classes_tiny.' + train_ext)
tgt_class_valid = os.path.join(tgt_dir, 'class/valid', 'classes_tiny.' + valid_ext)
tgt_sim_train = os.path.join(tgt_dir, 'sim/train', 'sim_tiny.' + train_ext)
tgt_sim_valid = os.path.join(tgt_dir, 'sim/valid', 'sim_tiny.' + valid_ext)
tgt_real_train = os.path.join(tgt_dir, 'real/train', 'real_tiny.' + train_ext)
tgt_real_valid = os.path.join(tgt_dir, 'real/valid', 'real_tiny.' + valid_ext)

tgt = [tgt_class_train, tgt_class_valid, tgt_sim_train, tgt_sim_valid, tgt_real_train, tgt_real_valid]

N = 128
for src_file, tgt_file in zip(src, tgt):
    src_data = np.load(src_file)
    np.save(tgt_file, src_data[:N])

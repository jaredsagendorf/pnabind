#!/usr/bin/env python

import sys
from pickle import dump
from geobind.nn.utils import loadDataset

data_dir = sys.argv[1] # directory where mesh data is located
data_files = sys.argv[2] # list of mesh data files
scaler_out = sys.argv[3] # name of output file

# load dataset
datafiles = [_.strip() for _ in open(data_files).readlines()]
dataset, transforms, info = loadDataset(datafiles, 2, "Y", data_dir,
    cache_dataset=False,
    balance='unmasked',
    remove_mask=False,
    scale=True,
    scaler=None
)

# pickle the scaler object
scaler = transforms["scaler"]
dump(scaler, open(scaler_out, "wb"))

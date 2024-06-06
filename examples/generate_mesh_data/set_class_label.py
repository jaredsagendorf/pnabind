import argparse
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("data_files", help="list of .npz files")
arg_parser.add_argument("sequence_info", help="information about structures to determine label")
arg_parser.add_argument("label_values", help="JSON file to map binding function to label values")
arg_parser.add_argument("-p", "--path", default=".", help="path where data files will be modified in-place")
arg_parser.add_argument("-y", "--label_key", default="Y", help="key to store label in data file")
ARGS = arg_parser.parse_args()

import numpy as np
from os.path import join as ospj
import json

# load sequence info
sequence_info = {info[0]: info for info in [_.strip().split(',') for _ in open(ARGS.sequence_info)]}

# read label value map
with open(ARGS.label_values) as FH:
    label_map = json.load(FH)

# loop over list of datafiles
for df in open(ARGS.data_files):
    df = df.strip()
    arrays = dict(np.load(ospj(ARGS.path, df), allow_pickle=True))
    
    seq_id = '_'.join(df.split('_')[0:2]) # this line will need to be modified depending on how datafiles/sequences are named!!
    c = label_map[sequence_info[seq_id][1]]
    arrays[ARGS.label_key] = c
    
    # save to file
    np.savez_compressed(df, **arrays)

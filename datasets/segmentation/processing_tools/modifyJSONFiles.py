#!/usr/bin/env python
import argparse
import json
import re

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--delete_objects_from", nargs='+')
arg_parser.add_argument("--delete_object_keys", nargs='+')
arg_parser.add_argument("--merge_files", nargs=2)
ARGS = arg_parser.parse_args()

def loadFile(file_name):
    with open(file_name) as FH:
        data = json.load(FH)
    return data

def writeFile(file_name, data):
    with open(file_name, "w") as FH:
        FH.write(json.dumps(data, indent=2))

if ARGS.merge_files is not None:
    data1 = loadFile(ARGS.merge_files[0])
    data2 = loadFile(ARGS.merge_files[1])
    data1.update(data2)
    
    # write out file
    writeFile(ARGS.merge_files[0], data1)

if ARGS.delete_objects_from is not None:
    for file_name in ARGS.delete_objects_from:
        data = loadFile(file_name)
        
        # find keys to remove
        delete = []
        for key in data:
            for del_key in ARGS.delete_object_keys:
                if re.search(del_key, key):
                    delete.append(key)
        
        # remove keys
        for key in delete:
            del data[key]
        
        writeFile(file_name, data)

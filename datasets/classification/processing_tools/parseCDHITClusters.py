#!/usr/bin/env python
import argparse
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("cluster_file", help="clustering output from CD-HIT")
arg_parser.add_argument("seq_info_file", help="information about sequences to use for selecting representatives")
arg_parser.add_argument("--num_per_cluster", default=1, type=int, help="number of sequences to sample per cluster")
arg_parser.add_argument("-o", "--output", default="clustered_ids", help="output file prefix")
arg_parser.add_argument("--sort_indices", nargs="+", type=int, default=[1], help="sort indices used to choose cluster representatives")
arg_parser.add_argument("--info_map", help="map values in info to sortable values")
ARGS = arg_parser.parse_args()

import re
import operator
import json

# read in info map if given
if ARGS.info_map:
    with open(ARGS.info_map) as FH:
        INFO_MAP = json.load(FH)
else:
    INFO_MAP = {}

# read in sequence info
SEQ_INFO = {}
for line in open(ARGS.seq_info_file):
    line = line.strip().split(',')
    SEQ_INFO[line[0]] = [INFO_MAP.get(_, _) for _ in line]

# read in clusters
count = 0
i = 0
CLST = open(ARGS.cluster_file).readlines()
CLUSTERS = []
while i < len(CLST):
    cluster = {
        "num": count,
        "ids": [],
        "size": 0,
    }
    i += 1
    while i < len(CLST):
        line = CLST[i]
        if line[0] == '>':
            break
        match = re.match('\d+\s+\d+aa, >sp\|([0-9a-zA-Z]+)\|', line)
        seqid = match.group(1)
        cluster["ids"].append(SEQ_INFO[seqid])
        cluster["size"] += 1
        i += 1
    CLUSTERS.append(cluster)
    count += 1

# write out clusters
OUT1 = open("%s.csv" % ARGS.output, "w")
OUT2 = open("%s.txt" % ARGS.output, "w")
for cluster in CLUSTERS:
    cluster['ids'].sort(key=operator.itemgetter(*ARGS.sort_indices), reverse=True)
    for seq in cluster['ids'][0:ARGS.num_per_cluster]:
        OUT1.write("{},{}\n".format(cluster["num"], seq[0]))
        OUT2.write("{}\n".format(seq[0]))
OUT1.close()
OUT2.close()



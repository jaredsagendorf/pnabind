#!/usr/bin/env python
import re
import sys

cluster_file = sys.argv[1]
max_per_cluster = int(sys.argv[2])

# Get preferred ids for cluster representatives
preferred = set()
if len(sys.argv) == 4:
    with open(sys.argv[3]) as FH:
        preferred.update([_.strip() for _ in FH])

# Read in clusters
count = 0
i = 0
CLST = open(cluster_file).readlines()
CLUSTERS = []
while i < len(CLST):
    cluster = {
        "num": count,
        "ids": set(),
        "size": 0,
    }
    i += 1
    while i < len(CLST):
        line = CLST[i]
        if line[0] == '>':
            break
        match = re.match('\d+\s+\d+aa, >([0-9a-zA-Z]+)', line)
        seqid = match.group(1)
        cluster["ids"].add(seqid)
        cluster["size"] += 1
        i += 1
    CLUSTERS.append(cluster)
    count += 1

# write out clusters
OUT1 = open("clustered_ids.csv", "w")
OUT2 = open("clustered_ids.txt", "w")
for cluster in CLUSTERS:
    # ordered so preferred sequences are first
    sequences = list(cluster['ids'] & preferred) + list(cluster['ids'] - preferred)
    for seqid in sequences[0:max_per_cluster]:
        OUT1.write("{},{}\n".format(cluster["num"], seqid))
        OUT2.write("{}\n".format(seqid))
OUT1.close()
OUT2.close()



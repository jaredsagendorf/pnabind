#!/usr/bin/env python
import sys
from itertools import takewhile
import networkx as nx

def parseKWData(file_name, G):
    # read in uniprot kw data
    with open(file_name) as handle:
        
        for line in handle:
            if line[0:2] != "ID":
                # keep going
                continue
            
            # enter into inner loop
            kw = line.strip('\n.').split()[1]
            for line in handle:
                if line == "//":
                    # reached end of this entry
                    break
                
                if line[0:2] == "HI":
                    line = line.strip('\n.').split(':')
                    path = [_.strip() for _ in line[1].split(';')]
                    nx.add_path(G, path)

sources = [
    "Nucleotide-binding",
    "ATP synthesis",
    "Activator",
    "Chromatin regulator",
    "DNA invertase",
    "DNA replication inhibitor",
    "DNA-binding",
    "Excision nuclease",
    "Initiation factor",
    "Ligase",
    "GTPase activation",
    "Guanine-nucleotide releasing factor",
    "Mobility protein",
    "Viral nucleoprotein",
    "Repressor",
    "Ribonucleoprotein",
    "RNA-binding",
    "Sigma factor",
    "Suppressor of RNA silencing",
    "Homeobox",
    "cAMP",
    "cGMP",
    "NAD",
    "NADP",
    "c-di-GMP"
]
kw_graph = nx.DiGraph()
parseKWData(sys.argv[1], kw_graph)

terms = set()
for source in sources:
    terms.add(source)
    terms.update(nx.descendants(kw_graph, source))
OUT = open("na_kw_terms.txt", "w")
for term in terms:
    OUT.write("%s\n" % term)
OUT.close()

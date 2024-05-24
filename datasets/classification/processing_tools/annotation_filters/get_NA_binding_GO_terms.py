#!/usr/bin/env python
import json
import urllib.request

seed_terms = [
    "GO:0000166", # nucleotide binding
    "GO:0001882", # nucleoside binding
    "GO:0003676", # nucleic acid binding
    "GO:0090304", # nucleic acid metabolic process
    "GO:0006974", # cellular response to DNA damage stimulus
    "GO:0022613", # ribonucleoprotein complex biogenesis
    "GO:0010467", # gene expression
    "GO:0140640", # catalytic activity, acting on a nucleic acid
    "GO:0051276", # chromosome organization
    "GO:0009117", # nucleotide metabolic process
    "GO:0017111"  # nucleoside-triphosphatase activity
]
terms_set = set(seed_terms)

def get_children(term, terms_set, depth=0, max_depth=3):
    if depth > max_depth:
        return
    
    url = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{}/complete".format(term)
    contents = urllib.request.urlopen(url).read()
    data = json.loads(contents)
    
    if "children" in data["results"][0]:
        for child in data["results"][0]["children"]:
            terms_set.add(child["id"])
            get_children(child["id"], terms_set, depth=depth+1, max_depth=max_depth)
    
    return

for term in seed_terms:
    print("Crawling for {}".format(term)) 
    get_children(term, terms_set, max_depth=5)

terms = open("na_go_terms.txt", "w")
for term in terms_set:
    terms.write("{}\n".format(term))
terms.close()

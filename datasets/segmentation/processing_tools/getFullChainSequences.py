#!/usr/bin/env python
import argparse
import requests
import json
from string import Template

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("query_file")
arg_parser.add_argument("chain_ids")
arg_parser.add_argument("--output_uniprot", action="store_true")
arg_parser.add_argument("--chain_ids_format", default="json", choices=["json", "list"])
arg_parser.add_argument("--remove_assembly_chain_ids", action="store_true")
ARGS = arg_parser.parse_args()

# Get list of valid PDBids
CHAIN_IDS = {}
if ARGS.chain_ids_format == "json":
    with open(ARGS.chain_ids) as FH:
        chain_mappings = json.load(FH)
    for assembly_file in chain_mappings:
        pdbid, _ = assembly_file.split('_')
        if pdbid not in CHAIN_IDS:
            CHAIN_IDS[pdbid] = set()
        for chain_id in chain_mappings[assembly_file].values():
            CHAIN_IDS[pdbid].add(chain_id)
else:
    chain_mappings = {}
    with open(ARGS.chain_ids) as FH:
        for line in FH:
            pdbid, chain_id = line.strip().split('_')
            chain_mappings["%s_%s.pdb" % (pdbid, chain_id)] = {chain_id: chain_id}
            if pdbid not in CHAIN_IDS:
                CHAIN_IDS[pdbid] = set()
            CHAIN_IDS[pdbid].add(chain_id)
PDBIDS = list(CHAIN_IDS.keys())

# Perform GraphQL query
graphql_url = "https://data.rcsb.org/graphql"
query = open(ARGS.query_file).read()
query = Template(query)
QUERY = {
    "query": query.substitute(entry_ids_array=','.join(['"%s"' %s for s in PDBIDS]))
}
print("Performing GraphQL request at {}".format(graphql_url))
req = requests.post(graphql_url, json=QUERY)
ENTRY_DATA = req.json()

if ARGS.output_uniprot:
    AUTH_ASYM_CHAIN_UNIPROT_MAPPINGS = open("chain_uniprot_mappings.txt", "w")

# Get Sequences
AUTH_ASYM_CHAIN_SEQ_MAPPINGS = {}
for entry in ENTRY_DATA["data"]["entries"]:
    pdbid = entry["rcsb_id"].lower()
    for polymer_entity in entry["polymer_entities"]:
        ptype = polymer_entity["entity_poly"]["rcsb_entity_polymer_type"]
        if ptype != "Protein":
            continue
        seq = polymer_entity["entity_poly"]["pdbx_seq_one_letter_code_can"]
        if polymer_entity["uniprots"]:
            uniprot = polymer_entity["uniprots"][0]["rcsb_id"]
        else:
            uniprot = "-"
        
        for polymer_instance in polymer_entity["polymer_entity_instances"]:
            auth_asym_id = polymer_instance["rcsb_polymer_entity_instance_container_identifiers"]["auth_asym_id"]
            key = "%s_%s" % (pdbid, auth_asym_id)
            
            AUTH_ASYM_CHAIN_SEQ_MAPPINGS[key] = seq
            if ARGS.output_uniprot and auth_asym_id in CHAIN_IDS[pdbid]:
                AUTH_ASYM_CHAIN_UNIPROT_MAPPINGS.write("%s %s\n" % (key, uniprot))

# create mappings
CHAIN_SEQ_MAPPINGS = {}
for assembly_file in chain_mappings:
    pdbid, _ = assembly_file.split('_')
    CHAIN_SEQ_MAPPINGS[assembly_file] = {}
    for chain_id in chain_mappings[assembly_file]:
        key = "%s_%s" % (pdbid, chain_mappings[assembly_file][chain_id])
        if key in AUTH_ASYM_CHAIN_SEQ_MAPPINGS:
            CHAIN_SEQ_MAPPINGS[assembly_file][chain_id] = AUTH_ASYM_CHAIN_SEQ_MAPPINGS[key]

FH = open("chain_sequence_mappings.json", "w")
FH.write(json.dumps(CHAIN_SEQ_MAPPINGS, indent=2))
FH.close()

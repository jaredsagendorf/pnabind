import json
import sys

json_file = sys.argv[1]
keys_file = sys.argv[2]

with open(json_file) as FH:
    D = json.load(FH)

keys = [_.strip() for _ in open(keys_file).readlines()]
Dnew = {key:D[key] for key in keys}

FH = open(json_file, "w")
FH.write(json.dumps(Dnew))
FH.close()

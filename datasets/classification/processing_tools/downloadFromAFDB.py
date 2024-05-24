import sys
import wget
from tqdm import tqdm
import os

not_found = open("404.txt","w")
for line in tqdm([l.strip() for l in open(sys.argv[1],'r').readlines()]):
    line = line.strip()
    url = "https://alphafold.ebi.ac.uk/files/AF-" + line + "-F1-model_v2.pdb"
    try: 
        filename = wget.download(url)
        os.rename("./AF-" + line + "-F1-model_v2.pdb", "./" + line + ".pdb")
    except Exception as e:
        not_found.write(line + "\n")
not_found.close()


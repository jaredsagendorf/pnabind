## Availability
To download pre-processed datasets, visit https://doi.org/10.5281/zenodo.11288475

## Benchmark Datasets
The following benchmark datasets are used in our manuscript for training and evaluating binding site prediciton models.

| Training Set | Corresponding Test Sets |
| ----------- | ----------- |
| DNA-573<sup>1</sup> | DNA-129<sup>1</sup>, DNA-181<sup>2</sup> |
| RNA-495<sup>1</sup> | RNA-117<sup>1</sup> |
| RBP09<sup>3</sup> | RPB61<sup>3</sup>  |

References:
1. Xia, Ying, et al. Nucleic acids research 49.9 (2021): e51.
2. Yuan Q, Chen S, Rao J, Zheng S, Zhao H, Yang Y. Brief Bioinform. 2022 Mar 10;23(2):bbab564.
3. Pengpai Li, Zhi-Ping Liu, Bioinformatics, Volume 38, Issue 8, March 2022, Pages 2162–2168.

## Steps for pre-processing structural data with provided tools
The following is a more or less complete guide for pre-processing the native bound protein datasets we used. Tools can be found in the directory `processing_tools`


1. First obtain list of PDB identifiers w/ chain ids, e.g.:	
	```
	cat chain_ids.txt	
	1cma_A
	1jgg_B
	3rca_C
	...
	```
2. Extract PDB ids and combine as a comma-separated list:
	```
	cut -d '_' -f 1 chain_ids.txt | sort | uniq | tr '[:upper:]' '[:lower:]' | sed -z 's/\n/,/g;s/,$/\n/' > pdbids.txt
 	```
4. Download biological assemblies using `download_assemblies.sh`:
	```
	download_assmblies.sh -f pdbids.txt -o <raw_structures_dir> -m 10
 	```
6. Run `findAssemblyFiles.py` to determine which biological assemblies contain the desired chains: 
	```
	findAssembyFiles.py chain_ids.txt <raw_structures_dir> --assembly_file_format=<pdb>
 	```
8. Manually resolve multiple assemblies for each chain (e.g., `1cma_A,1cma.pdb1,1cma.pdb2 --> 1cma_A,1cma.pdb2`). This occurs whenever a chain appears in more than one biological assembly.
9. Extract relevant protein chains from assemblies using `extractChains.py`: 
	```
	extractChains.py chain_assemblies.txt <raw_structures_dir> <structures_dir> --keep_nucleotide_chains --assembly_file_format=<pdb> --split_large_structures --split_structure_threshold=45000 --remove_disconnected_components --min_chain_length=30
 	```
	*WARNING*: test sets should be processed separate from training sets to avoid test and training chains being merged into same structure!
11. Get chain masks using 'getChainMasks.py'. This will mask out any chains included in the structure which are *NOT*  listed in the original `chain_ids.txt` file. 
	```
	getChainMasks.py <chain_info_file>
 	```
13. Get full chain sequences using `getFullChainSequences.py`: 
	```
	python getFullChainSequences.py <graphql query file> chain_mappings.json
 	```
15. Get binding site residues using `getBindingResidueIds.py`:
	```
	python getBindingResidueIds.py <structure_dir> <label FASTA file> -m chain_mappings.json
 	```
## Steps for generating mesh data
See "generate_mesh_data" in examples.

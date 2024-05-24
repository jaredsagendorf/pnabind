## Availability
To download pre-processed datasets, visit https://doi.org/10.5281/zenodo.11288475

## Steps for creating datasets
The following is a more or less complete guide for generating the AlphaFold datasets we used. Some steps may require a lot of data processing and are best done in parallel.

1. The files `swissprot_dna_binding.tab`, `swissprot_rna_binding.tab` and `swissprot_other.tab` were downloaded from UniProt with the following criteria:
	swissprot_dna_binding.tab: reviewed = yes AND key words = 'DNA-binding'
	swissprot_dna_binding.tab: reviewed = yes AND key words = 'RNA-binding'
	swissprot_other.tab: reviewed = yes AND key words = NOT 'DNA-binding' AND kew words = NOT 'RNA-binding'

	These can be updated if creating a new dataset.

2. Additional filters were applied using the script `filterTableByAnnotations.py`:
	```
	SCRIPT_DIR=../processing_tools
	python $SCRIPT_DIR/filterTableByAnnotaions.py swissprot_rna_binding.tab clustering/rna_binding_info -a available_in_AFDB.txt -K 'DNA-binding' --max_length 800 --min_annotation_score 2 --label 'rna_binding' > clustering/rna_binding_1.txt	
	python $SCRIPT_DIR/filterTableByAnnotaions.py swissprot_dna_binding.tab clustering/dna_binding_info -a available_in_AFDB.txt -K 'RNA-binding' --max_length 800 --min_annotation_score 2 --label 'dna_binding' > clustering/dna_binding_1.txt	
	python $SCRIPT_DIR/filterTableByAnnotaions.py swissprot_other.tab clustering/non_binding_info -a available_in_AFDB.txt -g $SCRIPT_DIR/annotation_filters/na_go_terms.txt -k $SCRIPT_DIR/annotation_filters/na_kw_terms.txt --max_length 800 --min_annotation_score 2 --label 'other' > clustering/non_binding_1.txt
	```
3. Sequences for each of the above lists were downloaded from UniProt using web API. Each list was then clustered by 70% sequence identity using CD-HIT, and one representative sequence was chosen for each cluster.
	```	
	cd-hit -i dna_binding_1.fasta -o dna_binding -c 0.7
	cd-hit -i rna_binding_1.fasta -o rna_binding -c 0.7	
	cd-hit -i non_binding_1.fasta -o non_binding -c 0.7

	python $SCRIPT_DIR/parseCDHITClusters.py dna_binding.clstr dna_binding_info.csv -o dna_binding_70
	python $SCRIPT_DIR//parseCDHITClusters.py rna_binding.clstr rna_binding_info.csv -o rna_binding_70
	python $SCRIPT_DIR//parseCDHITClusters.py non_binding.clstr non_binding_info.csv -o non_binding_70
	```
4. Alphafold structures were then downloaded for the remaining sequences and assessed for quality using the script `removeLowConfidenceResidues.py`:
	```	
	cat dna_binding_70.txt rna_binding_70.txt non_binding_70.txt > sequences_70.txt
	mkdir raw_structures
	mkdir structures
	cd raw_structures	
	python $SCRIPT_DIR/downloadFromAFDB.py sequences_70.txt
	ls *.pdb > ../raw_structure_list.txt # this can be broken up for batch processing
	cd ..
	python $SCRIPT_DIR/removeLowConfidenceResidues.py raw_structure_list.txt raw_structures structures
	```
5. Sequences were filtered based on the following structural criteria:
	```
	awk -F ','  'BEGIN {OFS=","} { if ( $3 == "1" && $4 > 0.5)  print $1 }' structure_stats.csv | grep -Ff dna_binding_70.txt > dna_binding_filtered.txt
	awk -F ','  'BEGIN {OFS=","} { if ( $3 == "1" && $4 > 0.5)  print $1 }' structure_stats.csv | grep -Ff rna_binding_70.txt > rna_binding_filtered.txt
	awk -F ','  'BEGIN {OFS=","} { if ( $3 == "1" && $4 > 0.5)  print $1 }' structure_stats.csv | grep -Ff non_binding_70.txt > non_binding_filtered.txt
	```
6. The remaining sequences were clustered again using PSI-BLAST at 35% sequence identity threshold to get the final list of sequences
## Steps for generating mesh data
TBD

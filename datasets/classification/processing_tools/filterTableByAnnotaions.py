import argparse
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("uniprot_tsv", help="tab-separate value table of uniprot accessions")
arg_parser.add_argument("output_file", help="CSV file to write to")
arg_parser.add_argument("alphafold_accessions", help="list of alphafold accession ids")
arg_parser.add_argument("-g", "--go_terms", help="file of excluded GO terms")
arg_parser.add_argument("-k", "--kw_terms", help="file of excluded uniprot kws")
arg_parser.add_argument("-G", "--go_terms_list", nargs="+", help="list of excluded GO terms")
arg_parser.add_argument("-K", "--kw_terms_list", nargs="+", help="list of excluded uniprot kws")
arg_parser.add_argument("--label", help="write given label in output")
arg_parser.add_argument("--max_length", default=1000, type=int, help="maximum sequence length")
arg_parser.add_argument("--min_annotation_score", default=1, type=int, help="minimum annotation score")
ARGS = arg_parser.parse_args()

# load alphafold accessions
ALPHAFOLD_ACCESSIONS = set([accession.strip() for accession in open(ARGS.alphafold_accessions)])

# load excluded terms
GO_TERMS = set()
KW_TERMS = set()
if ARGS.go_terms:
    GO_TERMS.update([l.strip() for l in open(ARGS.go_terms)])
if ARGS.kw_terms:
    KW_TERMS.update([l.strip() for l in open(ARGS.kw_terms)])
if ARGS.go_terms_list:
    GO_TERMS.update(ARGS.go_terms_list)
if ARGS.kw_terms_list:
    KW_TERMS.update(ARGS.kw_terms_list)

# loop over uniprot tsv file and keep all rows matching criteria
filtered = []
FH = open(ARGS.uniprot_tsv); next(FH) # skip header
OUT = open("%s.csv" % ARGS.output_file, "w")
for line in FH:
    fields = line.strip('\n').split("\t")
    
    # check length
    length = int(fields[5])
    if length > ARGS.max_length:
        continue
    
    # check annotation score
    score = int(fields[6][0])
    if score < ARGS.min_annotation_score:
        continue
    
    # check if alphafold structure exists
    accession = fields[0]
    if accession not in ALPHAFOLD_ACCESSIONS:
        continue
    
    # check for presense of forbidden keywords
    gos = set([_.strip() for _ in fields[8].split(";")])
    kws = set([_.strip() for _ in fields[10].split(";")])
    go_int = gos.intersection(GO_TERMS)
    kw_int = kws.intersection(KW_TERMS)
    
    if (len(go_int) > 0) or (len(kw_int) > 0):
        continue
    
    # keep accession
    print(accession)
    if ARGS.label:
        OUT.write("%s,%d,%d,%s\n" % (accession, score, length, ARGS.label))
    else:
        OUT.write("%s,%d,%d\n" % (accession, score, length))
OUT.close()

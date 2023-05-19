import os
import tempfile
import subprocess
import numpy as np

class ChainSequence(object):
    def __init__(self, chain, full_seq=None):
        from Bio.SeqUtils import seq1
        
        self.chain_seq = seq1(''.join([residue.resname for residue in chain]))
        self.seq_map = {}
        if full_seq:
            self.seq = full_seq
            
            # align sequence in chain to full seq to obtain mapping
            aln = alignSequences(self.seq, self.chain_seq)
            self.aln = aln
            
            # Add mappings
            residues = [residue for residue in chain]
            map_full_to_chain = aln["alignment"]["map_1to2"]
            for i in range(len(full_seq)):
                if map_full_to_chain[i] is not None:
                    ri = map_full_to_chain[i]
                    rid = residues[ri].get_full_id()
                    self.seq_map[rid] = i
        else:
            self.seq = self.chain_seq
            self.seq_map = {residue.get_full_id(): i for i, residue in enumerate(chain)}
        
    def __len__(self):
        return len(self.seq)
    
    def __getitem__(self, residue_id):
        return self.seq_map[residue_id]
    
    def writeFASTA(self, handle, header="HEAD", lw=60):
        handle.write(">{}\n".format(header))
        start = 0
        for i in range(len(self.seq) // lw + 1):
            slc = self.seq[start:start+lw]
            handle.write("{}\n".format(slc))
            start += lw

def alignSequences(seq1, seq2):
    from Bio import pairwise2
    from Bio.pairwise2 import format_alignment
    from Bio.Align import substitution_matrices
    
    BLOSUM = substitution_matrices.load("BLOSUM62")
    a = pairwise2.align.globalds(seq1, seq2, BLOSUM, -5.0, -0.5, one_alignment_only=True, penalize_end_gaps=False, penalize_extend_when_opening=True)
    
    seqa, matches, seqb = format_alignment(*a[0]).split("\n")[0:3]
    
    # Get residue ids corresponding to aligned labels
    ia = 0
    ib = 0
    map_a2b = [None]*len(seqa)
    map_b2a = [None]*len(seqb)
    for i in range(len(matches)):
        if seqa[i] == '-':
            ib += 1
        
        if seqb[i] == '-':
            ia += 1
        
        if matches[i] == '|' or matches[i] == '.':
            map_a2b[ia] = ib
            map_b2a[ib] = ia
            ib += 1
            ia += 1
    
    aln = {
        "seq1": seq1,
        "seq2": seq2,
        "alignment": {
            "seq1": seqa,
            "seq2": seqb,
            "matches": matches,
            "map_1to2": map_a2b,
            "map_2to1": map_b2a
        }
    }
    
    return aln

def runPSIBLAST(query_file, db_name, db_dir, 
        executable="psiblast", num_iterations=3, 
        ascii_pssm="pssm.txt", quiet=True,
        evalue=10, gapopen=11, gapextend=1,
        matrix="BLOSUM62", num_threads=12,
        word_size=2
    ):
    # set up environment variables
    my_env = os.environ.copy()
    my_env["BLASTDB"] = db_dir
    
    # execute PSIBLAST
    psi_args = {
        "-query": query_file,
        "-db" :db_name,
        "-num_iterations": num_iterations,
        "-out_ascii_pssm": ascii_pssm,
        "-evalue": evalue,
        "-gapopen": gapopen,
        "-gapextend": gapextend,
        "-matrix": matrix,
        "-inclusion_ethresh": 0.001,
        "-num_threads": num_threads,
        "-word_size": word_size
    }
    args = [executable, "-save_pssm_after_last_round"]
    for key, val in psi_args.items():
        args.append(key)
        args.append(str(val))
    
    if quiet:
        FNULL = open(os.devnull, 'w')
        subprocess.call(args, stdout=FNULL, stderr=FNULL, env=my_env)
        FNULL.close()
    else:
        subprocess.call(args, env=my_env)
    
    return ascii_pssm

def parsePSSM(file_name, fmt="ascii"):
    def _from_ascii(pssm_file):
        file_content = open(pssm_file, 'r').readlines()
        values_list = []
        for line in file_content[3:]:
            if line == "\n":
                break
            values_list.append(np.array(" ".join(line[:-1].split()).split(' ')[2:22]))
            
        return np.array(values_list, dtype=np.float32)
    
    if fmt == "ascii":
        return _from_ascii(file_name)
    else:
        pass

def parseHHM(file_name, fmt="ascii"):
    def _from_ascii(hhm_file):
        file_content = open(hhm_file, 'r').readlines()
        ptr = 0
        while file_content[ptr][0] != "#":
            ptr +=1 
        ptr += 5
        
        values_list = []
        for i in range(ptr, len(file_content), 3):
            if file_content[i] == "//\n":
                continue
            record = file_content[i].strip().split()[2:-1]
            record += file_content[i+1].strip().split()
            for i in range(len(record)):
                if record[i] == "*":
                    record[i] = 99999
                else:
                    record[i] = int(record[i])
            freq = [2**(-x/1000) for x in record[0:27]]
            divs = [0.001*x for x in record[-3:]]
            values_list.append(freq+divs)
        
        return np.array(values_list, dtype=np.float64)
    
    if fmt == "ascii":
        hhm = _from_ascii(file_name)
    else:
        pass
    
    return hhm

def runHHBLITS(query_file, db_name, db_dir,
        executable="hhblits", cpu=8, hhm_file="model.hhm",
        quiet=True
    ):
    # execute HHBLITS
    args = [executable, "-v", "0", "-Z", "0" "-hide_cons", "-hide_pred"]
    blits_args = {
        "-i": query_file,
        "-ohhm": hhm_file,
        "-d": os.path.join(db_dir, db_name),
        "-cpu": cpu
    }
    for key, val in blits_args.items():
        args.append(key)
        args.append(str(val))
    
    if quiet:
        FNULL = open(os.devnull, 'w')
        subprocess.call(args, stdout=FNULL, stderr=FNULL)
        FNULL.close()
    else:
        subprocess.call(args)
    
    return hhm_file

def getMSAFeatures(chain, 
        full_seq=None, seq_map=None,
        blast_db_name=None, blast_db_dir=None,
        blits_db_name=None, blits_db_dir=None,
        run_blast=True, run_blits=True,
        blast_kwargs={}, blits_kwargs={}
    ):
    blast_cols = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
    blits_cols = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y","MM","MI","MD","IM","II","DM","DD","Ne","Ne_I","Ne_D"]
    feature_names = []
    # create chain sequence object
    if seq_map is None:
        seq_map = ChainSequence(chain, full_seq=full_seq)
    
    # set up temp dir
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        
        # write FASTA file
        fasta_fh = open("tmp.fasta", "w")
        seq_map.writeFASTA(fasta_fh)
        fasta_fh.close()
        
        # run PSIBLAST
        if run_blast:
            pssm_file = runPSIBLAST("tmp.fasta", blast_db_name, blast_db_dir, **blast_kwargs)
            if os.path.exists(pssm_file):
                pssm = parsePSSM(pssm_file)
            else:
                pssm = np.zeros((len(seq_map), 20))
            feature_names += ["%s_%s" % ("pssm", cname) for cname in blast_cols]
        
        # run HHBLITS
        if run_blits:
            hhm_file = runHHBLITS("tmp.fasta", blits_db_name, blits_db_dir, **blits_kwargs)
            if os.path.exists(hhm_file):
                hhm = parseHHM(hhm_file)
            else:
                hhm = np.zeros((len(seq_map), 30))
            feature_names += ["%s_%s" % ("hmm", cname) for cname in blits_cols]
    os.chdir(cwd)
    
    # map MSA features to residues
    for residue in chain:
        rid = residue.get_full_id()
        i = seq_map[rid]
        if run_blast:
            for j in range(20):
                cname = blast_cols[j]
                fname = "%s_%s" % ("pssm", cname)
                for atom in residue:
                    atom.xtra[fname] = pssm[i, j]
        if run_blits:
            for j in range(30):
                cname = blits_cols[j]
                fname = "%s_%s" % ("hmm", cname)
                for atom in residue:
                    atom.xtra[fname] = hhm[i, j]
    
    return feature_names

import os
from rnapolis.annotator import extract_secondary_structure
from rnapolis.parser import read_3d_structure

from tqdm import tqdm

test_cifs_path = "/home/mjustyna/data/rna3db-mmcifs/test-300/"
drfold_seq_path = "/home/mjustyna/software/DRfold/validation"

cifs = os.listdir(test_cifs_path)
for cif in tqdm(cifs):
    path = os.path.join(test_cifs_path, cif)
    with open(path) as f:
        structure3d = read_3d_structure(f, 1)
        structure2d = extract_secondary_structure(structure3d, 1)
    fasta = structure2d.dotBracket.split('\n')[:2]
    fasta = "\n".join(fasta) +'\n'
    triple_fasta = fasta * 3
    # print(triple_fasta)

    fasta_name = cif.replace(".cif", ".fasta")
    with open(os.path.join(drfold_seq_path, fasta_name), 'w') as f:
        f.write(triple_fasta)

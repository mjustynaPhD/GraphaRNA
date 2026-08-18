import os
import shutil

data_dir = "../data/RNA-PDB-clean/test-pkl/"
pdb_dir = "../../data/non_rRNA_tRNA/"
out_dir = "../../data/RNA-GNN-test-pdb/"

pkl_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
os.makedirs(out_dir, exist_ok=True)
for f in pkl_files:
    pdb_name = f.replace('.pkl', '.pdb')
    pdb_path = os.path.join(pdb_dir, pdb_name)
    # copy the pdb file to the output directory
    shutil.copy(pdb_path, out_dir)
    print(f"Copied {pdb_name} to {out_dir}")

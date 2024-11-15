import os

PDB_DOWNLOAD = "https://files.rcsb.org/download/"
SAVE = "/home/mjustyna/data/eval_examples_pdb"
CLEAN_PDB = f"{SAVE}/clean/"
file = "eval_pdb_ids.txt"
with open(file, 'r') as f:
    lines = f.readlines()
lines = [l.strip() for l in lines if len(l.strip())>0 and not l.startswith('#')]
pdbs = [l.split('|')[0] for l in lines]
chains = [l.split('|')[2] for l in lines]
chains = [l.replace('Chain', '').strip() for l in chains]
print(pdbs)
print(chains)
print(len(pdbs))


for pdb, chain in zip(pdbs, chains):
    command = f"curl -o {SAVE}/{pdb}.pdb {PDB_DOWNLOAD}/{pdb}.pdb"
    if not os.path.exists(os.path.join(SAVE, pdb+".pdb")):
        os.system(command)
    else:
        print(pdb, "already exists. Skip download.")
    
    if not os.path.exists(CLEAN_PDB):
        os.makedirs(CLEAN_PDB)

    rna_tools = f"rna_pdb_tools.py --no-hr --extract-chain {chain} {SAVE}/{pdb}.pdb > {CLEAN_PDB}/{pdb}.pdb"
    print(rna_tools)
    os.system(rna_tools)

# rna_pdb_tools.py --no-hr --extract-chain B 2DR8.pdb

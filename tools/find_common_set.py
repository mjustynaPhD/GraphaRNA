import os

csv_file = "tools/HL_12078.6.csv"
target_files = "/home/mjustyna/data/rna3db-mmcifs/test-500"
with open(csv_file, 'r') as f:
    lines = f.readlines()

pdbs_ids = [line.split("|")[0].replace('"', '') for line in lines]
pdbs_ids = set(pdbs_ids)

targets = os.listdir(target_files)
targets_pdbs = set([pdb.split("_")[0].upper() for pdb in targets])
print(pdbs_ids.intersection(targets_pdbs))

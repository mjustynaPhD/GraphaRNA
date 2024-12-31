import os
import shutil

remove_path = "/home/mjustyna/data/full_PDB"

with open("tools/remove_pdbs.txt", "r") as f:
    ignored_ids = f.readlines()
    ignored_ids = [id.strip() for id in ignored_ids]

# remove_files = [f.replace('.pdb', '.pkl') for f in ignored_ids]
remove_files = ignored_ids
for f in remove_files:
    path = os.path.join(remove_path, f)
    if os.path.exists(path):
        os.remove(path)
    else:
        print(f"File {f} does not exist")
    pass

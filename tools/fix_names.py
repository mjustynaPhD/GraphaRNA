import os

path='/home/mjustyna/RNA-GNN/samples/super-mountain-68-rna3db-seed=4/715'
files = os.listdir(path)
for f in files:
    new_name = f.replace(".cif", "")
    os.rename(os.path.join(path, f), os.path.join(path, new_name))
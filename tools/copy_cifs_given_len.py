import os
import shutil
from rnapolis.annotator import extract_secondary_structure
from rnapolis.parser import read_3d_structure
from tqdm import tqdm


data_dir = '/home/mjustyna/data/rna3db-mmcifs/train_cifs/'
output_dir = '/home/mjustyna/data/rna3db-mmcifs/train-300/'

files = os.listdir(data_dir)
files = [f for f in files if f.endswith('.cif')]
os.makedirs(output_dir, exist_ok=True)

for f in tqdm(files):
    # if file size is bigger than 1MB, skip it
    if os.path.getsize(os.path.join(data_dir, f)) > 1000000:
        continue
    if os.path.exists(os.path.join(output_dir, f)):
        continue

    with open(os.path.join(data_dir, f), 'r') as file:
        structure3d = read_3d_structure(file, 1)
        if len(structure3d.residues) <= 300:
            shutil.copy(os.path.join(data_dir, f), os.path.join(output_dir, f))
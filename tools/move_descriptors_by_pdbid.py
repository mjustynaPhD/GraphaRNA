import os
import shutil
import pandas as pd
from tqdm import tqdm

DESCS = '/data/3d/input_data/desc-pdbs/'
SEG_DESCS = '/data/3d/input_data/descs/segments_2-output.txt'
OUT_DIR = '/data/3d/input_data/desc-grouped-pdb/'


def move_to_dirs(pdbs, out_dir):
    all_descs = os.listdir(DESCS)
    for desc in tqdm(all_descs):
        pdb_id = "_".join(desc.split('_')[:3])
        if pdb_id in pdbs:
            os.makedirs(os.path.join(out_dir, pdb_id), exist_ok=True)
            shutil.copy(os.path.join(DESCS, desc), os.path.join(out_dir, pdb_id, desc))           

def main():
    selected_pdbs = pd.read_csv('/data/3d/RNA-GNN/tools/selected-structures.csv')
    selected_pdbs = set(selected_pdbs['id'].values)
    with open(SEG_DESCS, 'r') as f:
        seg_desc_files = f.readlines()
    seg_desc_files = [f.split()[1] for f in seg_desc_files]
    seg_desc_files = set(["_".join(f.split('_')[:3]) for f in seg_desc_files])
    pdbs_intersect = selected_pdbs.intersection(seg_desc_files)
    # pdbs intersect to dict
    pdbs_intersect = {pdb: True for pdb in pdbs_intersect}
    move_to_dirs(pdbs_intersect, out_dir=OUT_DIR)
    pass


if __name__ == '__main__':
    main()
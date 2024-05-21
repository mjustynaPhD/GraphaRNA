# Run with: python run_dcompare.py /data/3d/input_data/descs/segments_3-output.txt

import sys
import os
from tqdm import tqdm
import multiprocessing
from functools import partial
import numpy as np

# FILES_PATH = '/data/3d/input_data/desc-pdbs/'
MASTER_PATH = '/data/3d/input_data/desc-grouped-pdb/'
FILES_PATH = '' # '/data/3d/input_data/desc-grouped-pdb/8HMZ_1_5-3/'
OUT_DIR = 'out2/'

def run_command(base1, base2, files_path):
    base1_pdb = base1 + ".pdb" if not base1.endswith('.pdb') else base1
    base2_pdb = base2 + ".pdb" if not base2.endswith('.pdb') else base2
    base1 = base1.replace('.pdb', '')
    base2 = base2.replace('.pdb', '')
    f1 = os.path.join(files_path, base1_pdb)
    f2 = os.path.join(files_path, base2_pdb)
    cmd = f"./dcompare.sh {f1} {f2} | grep 'are structurally similar' | tail -1 > {OUT_DIR}/{base1}-{base2}.txt"
    # print(cmd)
    # cmd = f"./dcompare.sh {f1} {f2} " #| grep 'are structurally similar'"
    if os.path.exists(f1) and os.path.exists(f2):
        os.system(cmd)
        # get size of the output file
        size = os.path.getsize(f'{OUT_DIR}/{base1}-{base2}.txt')
        return size>0
    else:
        return False

def read_list(list_file):
    with open(list_file) as f:
        lines = f.readlines()
    files = [f.split('\t')[1] for f in lines]
    return files

def main():
    # list_file = sys.argv[1]
    # files = read_list(list_file)
    dirs = os.listdir(MASTER_PATH)  
    print(f'Dirs: {len(dirs)}')
    # run it in parallel with multiprocessing
    # pool = multiprocessing.Pool(processes=6)
    for d in dirs:
        files_path = os.path.join(MASTER_PATH, d)
        files = os.listdir(files_path)
        print(f"Processing {files_path} with {len(files)} files.")
        with multiprocessing.Pool(processes=6) as pool:
            ignored = []
            for i in tqdm(range(0, len(files))):
                if i in ignored:
                    print(f"Ignoring {files[i]}")
                    continue
                f1 = files[i]
                rcmd = partial(run_command, f1, files_path=files_path)
                similar = np.array(pool.map(rcmd, files[i+1:]))
                ignored_ids = np.where(similar==True)[0]
                if ignored_ids.size > 0:
                    ignored_ids = ignored_ids + i + 1
                    ignored.extend(ignored_ids)

if __name__ == '__main__':
    main()
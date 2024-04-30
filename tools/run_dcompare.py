# Run with: python run_dcompare.py /data/3d/input_data/descs/segments_3-output.txt

import sys
import os
from tqdm import tqdm
import multiprocessing
from functools import partial

FILES_PATH = '/data/3d/input_data/desc-pdbs/'

def run_command(base1, base2):
    f1 = os.path.join(FILES_PATH, base1 + '.pdb')
    f2 = os.path.join(FILES_PATH, base2 + '.pdb')
    # cmd = f"./dcompare.sh {f1} {f2} | grep 'are structurally similar' | tail -1 > out/{base1}-{base2}.txt"
    cmd = f"./dcompare.sh {f1} {f2} | grep 'are structurally similar' > out/{base1}-{base2}.txt"
    if os.path.exists(f1) and os.path.exists(f2):
        os.system(cmd)

def read_list(list_file):
    with open(list_file) as f:
        lines = f.readlines()
    files = [f.split('\t')[1] for f in lines]
    return files

def main():
    list_file = sys.argv[1]
    files = read_list(list_file)
    # for f1 in tqdm(files[:9]):
    #     for f2 in files[1:10]:
    #         run_command(f1, f2)
    
    # run it in parallel with multiprocessing
    pool = multiprocessing.Pool(processes=8)
    for f1 in tqdm(files[:19]):
        rcmd = partial(run_command, f1)
        pool.map(rcmd, files[1:20])
        


if __name__ == '__main__':
    main()
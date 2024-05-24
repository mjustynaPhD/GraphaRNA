# Run with: python run_dcompare.py /data/3d/input_data/descs/segments_3-output.txt
import re
import sys
import os
from tqdm import tqdm
import multiprocessing
from functools import partial
import numpy as np

FILES_PATH = '/data/3d/input_data/desc-pdbs/'
OUT_DIR = 'tools/out2/'

def run_command(bases):
    base1, base2 = bases
    f1 = os.path.join(FILES_PATH, base1 + '.pdb')
    f2 = os.path.join(FILES_PATH, base2 + '.pdb')
    if os.path.exists(f'{OUT_DIR}/{base1}-{base2}.txt'):
        return os.path.getsize(f'{OUT_DIR}/{base1}-{base2}.txt') > 0
    elif base1 == base2:
        return False
    cmd = f"tools/dcompare.sh {f1} {f2} | grep 'are structurally similar' | tail -1 > {OUT_DIR}/{base1}-{base2}.txt"
    # cmd = f"./dcompare.sh {f1} {f2} " #| grep 'are structurally similar'"
    if os.path.exists(f1) and os.path.exists(f2):
        os.system(cmd)
        # get size of the output file
        size = os.path.getsize(f'{OUT_DIR}/{base1}-{base2}.txt')
        return size>0
    else:
        return False

def get_ignored_ids():
    tmp_files = os.listdir(OUT_DIR)
    ignore = {}
    for f in tmp_files:
        # file format is '1CSL_1_B-A_A_45_C-1CSL_1_B-A_A_46_G.txt', which is base1-base2.txt
        # find re to split into: 1CSL_1_B-A_A_45_C and 1CSL_1_B-A_A_46_G
        # generate re to match pdb ids.
    
        # names = re.findall('[A-Z0-9]{4}[_][A-Z0-9]{1,2}[_][A-Z_-]{3,22}[_][A-Z0-9]{1,4}[_][ACGU]', f)
        names = re.findall('[0-9][A-Z0-9]{3}[_][A-Z0-9]{1,2}[_]', f)
        assert len(names) == 2
        name1, name2 = names
        # name1, name2 = f.replace('.txt', '').split('-')
        id1 = f[1:].find(name2)
        name1 = f[:id1]
        name2 = f[id1+1:].replace('.txt', '')
        if os.path.getsize(f'{OUT_DIR}/{f}') > 0 and name1 != name2:
            ignore[name2] = True

    return ignore

def add_to_ignored(results, bases, ignore):
    ignore_ids = np.where(results==True)[0]
    for i in ignore_ids:
        ignore[bases[i][1]] = True
    return ignore

def main():
    list_file = "tools/descs-seg-2.csv"
    ignore = get_ignored_ids()
    print(f"Ignoring {len(ignore)}")
    with open(list_file) as f:
        bases=[]
        for i, line in enumerate(f):
            base1, base2 = line.strip().split(',')
            if base1 in ignore or base2 in ignore:
                continue
            bases.append((base1, base2))
            if len(bases) == 10:
                print(i)
                with multiprocessing.Pool(6) as p:
                    results = np.array(p.map(run_command, bases))
                ignore = add_to_ignored(results, bases, ignore)
                bases = []


if __name__ == '__main__':
    main()
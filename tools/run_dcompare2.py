# Run with: python run_dcompare.py /data/3d/input_data/descs/segments_3-output.txt

import sys
import os
from tqdm import tqdm
import multiprocessing
from functools import partial

FILES_PATH = '/data/3d/input_data/desc-pdbs/'

def run_command(bases):
    base1, base2 = bases
    f1 = os.path.join(FILES_PATH, base1 + '.pdb')
    f2 = os.path.join(FILES_PATH, base2 + '.pdb')
    cmd = f"./dcompare.sh {f1} {f2} | grep 'are structurally similar' | tail -1 > out/{base1}-{base2}.txt"
    # cmd = f"./dcompare.sh {f1} {f2} " #| grep 'are structurally similar'"
    if os.path.exists(f1) and os.path.exists(f2):
        os.system(cmd)
        # get size of the output file
        size = os.path.getsize(f'out/{base1}-{base2}.txt')
        return size>0
    else:
        return False

def main():
    list_file = sys.argv[1]
    with open(list_file) as f:
        bases=[]
        for i, line in enumerate(f):
            base1, base2 = line.strip().split(',')
            bases.append((base1, base2))
            if len(bases) == 100:
                print(i)
                with multiprocessing.Pool(6) as p:
                    p.map(run_command, bases)
                bases = []

        


if __name__ == '__main__':
    main()
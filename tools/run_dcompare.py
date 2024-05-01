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
    cmd = f"./dcompare.sh {f1} {f2} | grep 'are structurally similar' | tail -1 > out/{base1}-{base2}.txt"
    # cmd = f"./dcompare.sh {f1} {f2} " #| grep 'are structurally similar'"
    if os.path.exists(f1) and os.path.exists(f2):
        os.system(cmd)
        # get size of the output file
        size = os.path.getsize(f'out/{base1}-{base2}.txt')
        return size>0
    else:
        return False

def read_list(list_file):
    with open(list_file) as f:
        lines = f.readlines()
    files = [f.split('\t')[1] for f in lines]
    return files

def main():
    """
    Main Idea of Algorithm
    ----------------------
    A -> B
    B -> C
    C -> D

    This means A is similar to B, B is similar to C, and C is similar to D, so A is similar to D and we don't need to compare A with C and D.
    
    If A is similar to B, then we break the loop and start comparing from B (as we assume that all similar childs of B are similar to A).

    """

    list_file = sys.argv[1]
    files = read_list(list_file)
    f_visited = {}
    chain = {}

    # if A is similart to B and B is similar to C, then A is similar to C
    # algorithm:
    # start with first A, compare with all other B
    # if A is similar to B, then break. Add B to the chain of A
    # start from B, compare with all other C. if C is similar to B, then break. Add C to the chain of A and B
    
    max = len(files)
    for i in tqdm(range(len(files[:max-1]))): # iterate over all files
        if files[i] in f_visited: # skip files already visited
            continue
        i1 = i
        top_level_f1 = None
        while i1 < len(files[:max-1]): # try to find a chain (a sequence of similar files)
            f1 = files[i1]
            if top_level_f1 is None:
                top_level_f1 = f1
            f_visited[f1] = True # this file is already visited so we will skip it in the future iterations

            for i2 in range(i1+1, len(files[:max])):
                f2 = files[i2]
                if run_command(f1, f2):
                    chain[top_level_f1] = chain.get(top_level_f1, []) # add childs to the chain
                    chain[top_level_f1].append(f2)
                    f_visited[f2] = True # f2 is visited, because it is similar to f1, so there is no need to compare it with other files
                    print(f1, f2, i1, i2)
                    i1 = i2
                    break
            if i2 == len(files[:max])-1: # if chain is over, break
                break
            i1 += 1

    # for i in tqdm(range(0, len(files[:max-1]))):
    #     for j in range(i+1, len(files[:max])):
    #         f1 = files[i]
    #         f2 = files[j]
    #         run_command(f1, f2)

    
    
    # run it in parallel with multiprocessing
    # pool = multiprocessing.Pool(processes=8)
    # for i, f1 in tqdm(enumerate(files[:10])):
    #     rcmd = partial(run_command, f1)
    #     pool.map(rcmd, files[i:11])
        


if __name__ == '__main__':
    main()
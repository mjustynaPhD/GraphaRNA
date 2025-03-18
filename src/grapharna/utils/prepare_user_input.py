import argparse
import os
from tqdm import tqdm

from grapharna.preprocess_rna_pdb import construct_graphs


def read_dotseq_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    seq_segments = lines[1].strip().split()
    dot = [l.replace(" ", "").strip() for l in lines]
    name = dot[0].replace(">", "").strip()
    return name, dot[2], seq_segments
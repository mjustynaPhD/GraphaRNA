import os
from tqdm import tqdm

PDBS_PATH = "/home/mjustyna/data/desc-pdbs/"
TXTS = "/home/mjustyna/data/descs/"
SIMRNA_OUT_PATH = "/home/mjustyna/data/sim_desc/"
RUN_SIMRNA="SimRNA"


def process_lines(lines):
    lines = [l.split('\t') for l in lines]
    lines = [l for l in lines if len(l) == 7]
    lines = [(l[1], int(l[2]), l[6].replace(",", "").strip()) for l in lines if int(l[2]) <= 3 and int(l[4]) <= 60]
    for desc_id, _, seq in lines:
        if not os.path.exists(os.path.join(PDBS_PATH, desc_id + ".pdb")):
            continue
        with open(os.path.join(SIMRNA_OUT_PATH, desc_id + ".seq"), "w") as f:
            f.write(seq)

def main():
    os.makedirs(SIMRNA_OUT_PATH, exist_ok=True)
    for txt in tqdm(os.listdir(TXTS)):
        with open(TXTS + txt, "r") as f:
            lines = f.readlines()
            process_lines(lines)

if __name__ == "__main__":
    main()
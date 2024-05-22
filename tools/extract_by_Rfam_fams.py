import os
import shutil
import pandas as pd
from tqdm import tqdm

RFAM_FILE = "tools/Rfam.pdb"
RFAM_FAMILIES = "tools/Rfam_families.txt"
DESC_FILES = "/home/mjustyna/data/desc-pdbs/"
rRNA_SAVE = "/home/mjustyna/data/rRNA_tRNA/"
non_rRNA_SAVE = "/home/mjustyna/data/non_rRNA_tRNA/"


def main():
    # rfam_file = pd.read_csv(RFAM_FILE, sep="\t")
    # with open(RFAM_FAMILIES, "r") as f:
    #     rfam_families = f.readlines()
    # rfam_families = [x.strip() for x in rfam_families]
    # rfam_families = [x for x in rfam_families if not x.startswith("#") and x != ""]
    # # choose records where rfam_acc is in rfam_families
    # rfam_file = rfam_file[rfam_file["rfam_acc"].isin(rfam_families)]
    # # group by PDB ID and chains
    # pdb_ids_chains = rfam_file.groupby(["pdb_id", "chain"]).first().reset_index()
    # pdb_ids_chains = pdb_ids_chains[['pdb_id', 'chain']]
    # # transform to dict. each pdb_id is a key, and the value is a list of chains
    # pdb_ids_chains = pdb_ids_chains.groupby('pdb_id')['chain'].apply(list).to_dict()

    with open("tools/ribosomal_rna.ids", 'r') as f:
        rRNA = f.readlines()
    rRNA = rRNA[0].strip().split(',')
    with open("tools/trna.ids", 'r') as f:
        tRNA = f.readlines()
    tRNA = tRNA[0].strip().split(',')
    rRNA.extend(tRNA)
    # make rRNA hashable as a dict (pdb_id, True)
    rRNA = [(x, True) for x in rRNA]
    pdbs_dict = dict(rRNA)


    pdb_files = os.listdir(DESC_FILES)
    os.makedirs(rRNA_SAVE, exist_ok=True)
    os.makedirs(non_rRNA_SAVE, exist_ok=True)
    for f in tqdm(pdb_files):
        pdb_id = f.split("_")[0]
        pdb_chain = f.split("_")[2]
        if pdb_id in pdbs_dict: # and pdb_chain in pdb_ids_chains[pdb_id]:
            shutil.copyfile(os.path.join(DESC_FILES, f), os.path.join(rRNA_SAVE, f))
        else:
            shutil.copyfile(os.path.join(DESC_FILES, f), os.path.join(non_rRNA_SAVE, f))


    pass

if __name__ == "__main__":
    main()
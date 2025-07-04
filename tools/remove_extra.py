import os
from tqdm import tqdm

DATASET_PATH = "/home/mjustyna/RNA-GNN/data/eval-pdb-all/"

def main():
    ignored_ids_file = "clean-test.names" #"all_ignored.ids"
    with open(ignored_ids_file, "r") as f:
        ignored_ids = f.readlines()
        ignored_ids = set([id.strip()[2:] for id in ignored_ids])
    
    dirs = os.listdir(DATASET_PATH)
    for d in dirs:
        print("Removing from", d)
        dir_files = os.listdir(os.path.join(DATASET_PATH, d))
        for id in tqdm(dir_files):
            path = os.path.join(DATASET_PATH, d, id) # id+".pkl")
            # if os.path.exists(path):
            if id not in ignored_ids:
                print("File does not exist:", path)
                os.remove(path)
    pass


if __name__ == "__main__":
    main()
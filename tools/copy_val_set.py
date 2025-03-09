import os
import random
import shutil
from tqdm import tqdm

random.seed(0)
train_dir = '../data/full_PDB/train-pkl'
val_dir = '../data/full_PDB/val-pkl'

os.makedirs(val_dir, exist_ok=True)

train_files = os.listdir(train_dir)
val_files = random.sample(train_files, 250)

for file in tqdm(val_files):
    shutil.move(os.path.join(train_dir, file), os.path.join(val_dir, file))
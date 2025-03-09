import os
import shutil

rhofold_preds = "/home/mjustyna/software/DRfold/single_fastas"
out_path = "/home/mjustyna/RNA-GNN/samples/DRfold/1segment"

os.makedirs(out_path, exist_ok=True)
for pdb_pred in os.listdir(rhofold_preds):
    print(pdb_pred)
    file = os.path.join(rhofold_preds, pdb_pred, "DPR.pdb")
    target_name = f"{pdb_pred}.pdb"
    target = os.path.join(out_path, target_name)
    shutil.copy(file, target)
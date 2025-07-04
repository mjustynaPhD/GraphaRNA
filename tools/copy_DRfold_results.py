import os
import shutil

rhofold_preds = "/home/mjustyna/software/DRfold/preds_all"
out_path = "/home/mjustyna/RNA-GNN/samples/DRfold-all/1_segment"

os.makedirs(out_path, exist_ok=True)
for pdb_pred in os.listdir(rhofold_preds):
    # print(pdb_pred)
    file = os.path.join(rhofold_preds, pdb_pred, "DPR.pdb")
    if not os.path.exists(file):
        print(f"File {file} does not exist, skipping.")
        continue
    target_name = f"{pdb_pred}.pdb"
    target = os.path.join(out_path, target_name)
    shutil.copy(file, target)
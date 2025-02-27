import os
# import bio to read cif structure and save it as pdb
from Bio.PDB import MMCIFParser, PDBIO

preds = "/home/mjustyna/boltz/5_segment"
out_path = "/home/mjustyna/RNA-GNN/samples/boltz/5_segment"

os.makedirs(out_path, exist_ok=True)
for pdb_pred in os.listdir(preds):
    
    file = os.path.join(preds, pdb_pred, "DPR.pdb")
    target_name = f"{pdb_pred.replace('boltz_results_', '')}"
    print(target_name)
    pred_struct = os.path.join(preds, pdb_pred, 'predictions', target_name, f'{target_name}_model_0.cif',)
    parser = MMCIFParser()
    structure = parser.get_structure(target_name, pred_struct)
    io = PDBIO()
    io.set_structure(structure)
    io.save(os.path.join(out_path, f"{target_name}.pdb"))
    
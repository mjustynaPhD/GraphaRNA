import os

# read cif files and save them as pdb files
from Bio.PDB import MMCIFParser, PDBIO

GT_TARGETS = "/home/mjustyna/data/motifs/il_pdbs/"
GT_DEST = "/home/mjustyna/RNA-GNN/samples/af3_il_preds/targets/"
AF_PREDS = "/home/mjustyna/RNA-GNN/samples/af3_il_preds/cifs/"
AF_PDBS = "/home/mjustyna/RNA-GNN/samples/af3_il_preds/pdbs/"
MY_PREDS = "/home/mjustyna/RNA-GNN/samples/glowing-terrain-25-il/800/"
MY_PREDS_DEST = "/home/mjustyna/RNA-GNN/samples/af3_il_preds/ours/"

targets = os.listdir(GT_TARGETS)
targets_keys = [t.lower().replace('.pdb', '').replace('-', '_') for t in targets]
targets_dict = dict(zip(targets_keys, targets))
preds = os.listdir(AF_PREDS)
preds = [p.lower().replace('.cif', '') for p in preds]
my_preds = os.listdir(MY_PREDS)
my_preds = [p for p in my_preds if p.endswith('-000001_AA.pdb')]
my_preds_c = [p.replace('-000001_AA.pdb', '').replace('-', '_').lower() for p in my_preds]
my_preds_dict = dict(zip(my_preds_c, my_preds))

print(preds)


for p in preds:
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(p, os.path.join(AF_PREDS, p + '.cif'))
    name = p
    name = name.replace('fold_', '')
    name = name.replace('_model_0', '')
    out_name = targets_dict.get(name, None)
    print(name, name in targets_dict)
    assert out_name is not None, ValueError(f"Target pdb file for {name} not found")
    io = PDBIO()
    io.set_structure(structure)
    io.save(os.path.join(AF_PDBS, out_name))
    # copy target pdb file to the same directory
    os.system(f"cp {GT_TARGETS}/{out_name} {GT_DEST}/{out_name}")
    os.system(f'cp {MY_PREDS}/{my_preds_dict[name]} {MY_PREDS_DEST}/{my_preds_dict[name]}')

import os
import argparse
import math
from tqdm import tqdm
from Bio.PDB import PDBParser
from Bio.PDB import Superimposer
import pandas as pd
# ignore Bio python warnings
import numpy as np
import warnings
from Bio import BiopythonWarning
from pymol import cmd
import barnaba as bb
warnings.simplefilter('ignore', BiopythonWarning)
import subprocess

from rnapolis.annotator import extract_secondary_structure
from rnapolis.parser import read_3d_structure

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--preds-path', type=str, default=None, help='Path to the predictions directory (*.trafl files)')
    args.add_argument('--templates-path', type=str, default=None, help='Path to the pdb templates directory')
    args.add_argument('--targets-path', type=str, default=None, help='Path to the pdb targets directory')
    args.add_argument('--output-name', type=str, default="rmsd.csv", help='Name of the output file')
    args.add_argument('--sim_rna', type=str, default=None, help='Path to the simRNA executable')
    args.add_argument('--arena_path', type=str, default=None, help='Path to the arena executable')
    args.add_argument("--lddt-logs", type=str, default='lddt', help="Path to the lDDT logs directory")
    args.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    return args.parse_args()

def generate_pdbs_from_trafl(trafl_path, targets_path, sim_rna_path, overwrite=False, pdb_postfix='.pdb', out_postfix='-000001_AA.pdb'):
    trafls = os.listdir(trafl_path)
    trafls = [trafl for trafl in trafls if trafl.endswith('.trafl')]
    for t in tqdm(trafls):
        base_name = t.replace('.trafl', '')
        pdb_name = base_name + pdb_postfix
        out_name = base_name + out_postfix
        if not overwrite and os.path.exists(os.path.join(trafl_path, out_name)):
            continue
        
        if os.path.exists(f"{targets_path}/{pdb_name}"):
            os.system(f"./run_SimRNA_trafl_to_pdb.sh {sim_rna_path} {targets_path}/{pdb_name} {trafl_path}/{t}")
        else:
            print(f"Skipping {t} as the target pdb file: {targets_path}/{pdb_name} does not exist")

def generate_pdbs_AA(samples_path, arena_path, overwrite=False, out_postfix='_AA.pdb'):
    samples = os.listdir(samples_path)
    samples = [sample for sample in samples if sample.endswith('.pdb') and not sample.endswith(out_postfix)]
    for s in tqdm(samples):
        base_name = s.replace('.pdb', '')
        out_name = base_name + out_postfix
        if not overwrite and os.path.exists(os.path.join(samples_path, out_name)):
            continue
        os.system(f"./run_ARENA_pdb_to_aa.sh {arena_path} {samples_path}/{s} {samples_path}/{out_name}")

def extract_2d_structure(pdb_path):
    with open(pdb_path, 'r') as f:
        structure3d = read_3d_structure(f, 1)
        structure2d = extract_secondary_structure(structure3d, 1)
    s2d = structure2d.dotBracket.split('>strand')
    if len(s2d) > 1:
        s2d = [s.strip().split('\n')[-1] for s in s2d]
        return "".join(s2d)
    else:
        return structure2d.dotBracket.split('\n')[-1]

def get_inf(s_pred, s_gt):
    assert len(s_pred) == len(s_gt), ValueError("Length of the predicted and ground truth sequences should be the same")
    s_pred = [1 if c != '.' else 0 for c in s_pred]
    s_gt = [1 if c != '.' else 0 for c in s_gt]
    tp = sum([1 for i in range(len(s_pred)) if s_pred[i] == 1 and s_gt[i] == 1])
    fp = sum([1 for i in range(len(s_pred)) if s_pred[i] == 1 and s_gt[i] == 0])
    fn = sum([1 for i in range(len(s_pred)) if s_pred[i] == 0 and s_gt[i] == 1])
    ppv = tp / (tp + fp) if tp + fp > 0 else 0
    sty = tp / (tp + fn) if tp + fn > 0 else 0
    inf = math.sqrt(ppv * sty)
    return inf

def get_tmscore_and_gdts(trafl_path, targets_path, pdb, pdb_name):
    # run system command and store the output
    output = subprocess.run(
        f"TMscore {trafl_path}/{pdb} {targets_path}/{pdb_name}",
        shell=True,
        capture_output=True,
        text=True
    )
    tmscore, gdts = filter_tmscore_output(output.stdout)
    return tmscore, gdts

def filter_tmscore_output(output):
    lines = output.split('\n')
    tmscore = None
    gdts = None
    for line in lines:
        if line.startswith('TM-score'):
            tmscore = line
        if line.startswith('GDT-TS-score'):
            gdts = line

    if tmscore is not None:
        tmscore = float(tmscore.split()[2])
    else:
        tmscore = np.nan
    if gdts is not None:
        gdts = float(gdts.split()[1])
    else:
        gdts = np.nan
    return tmscore, gdts

def get_lddt(pred_pdb, ref_pdb, raw_pdb, lddt_logs):
    raw_pdb = raw_pdb.replace('.pdb', '')
    if not os.path.exists(lddt_logs):
        os.makedirs(lddt_logs)
    out_json = f"{lddt_logs}/{raw_pdb}_out.json"
    if not os.path.exists(out_json):
        lddt_command = f"ost compare-structures --bb-lddt -o {out_json} -m {pred_pdb} -r {ref_pdb}"
        # just run the command and don't capture the output
        output = subprocess.run(lddt_command, shell=True, capture_output=True, text=True)
    get_output_cmd = f'cat {out_json} | grep "bb_lddt" | head -n 1'
    lddt_output = subprocess.run(get_output_cmd, shell=True, capture_output=True, text=True)
    if lddt_output.stderr:
        print(f"Error in getting lDDT output: {lddt_output.stderr}")
        return np.nan
    lddt_value = lddt_output.stdout.split(":")[1].strip().replace(',', '')
    return float(lddt_value) if lddt_value else np.nan    

def superimpose_pdbs(trafl_path, targets_path, out_postfix='-000001_AA.pdb', method:str='pymol', lddt_logs:str='lddt'):
    pdbs = os.listdir(trafl_path)
    pdbs = [pdb for pdb in pdbs if pdb.endswith(out_postfix)]
    outs = []
    for pdb in tqdm(pdbs):
        base_name = pdb.replace(out_postfix, '')
        pdb_name = base_name + '.pdb' if not base_name.endswith('.pdb') else base_name
        if os.path.exists(f"{targets_path}/{pdb_name}"):
            ref_2d_structure = extract_2d_structure(f"{targets_path}/{pdb_name}")
            pred_2d_structure = extract_2d_structure(f"{trafl_path}/{pdb}")

            try:
                inf = get_inf(pred_2d_structure, ref_2d_structure)
            except Exception as e:
                print(f"Skipping {pdb} as the INF calculation failed")
                continue
            if method == 'pymol':
                rms = align_pymol(trafl_path, targets_path, pdb, pdb_name)
            elif method == 'biopython':
                rms = align_biopython(trafl_path, targets_path, pdb, pdb_name)
            try:
                ermsd = bb.ermsd(f"{targets_path}/{pdb_name}", f"{trafl_path}/{pdb}")[0]
            except Exception as e:
                print(f"Skipping {pdb} as the eRMSD calculation failed")
                ermsd = np.nan

            try:
                tmscore, gdts = get_tmscore_and_gdts(trafl_path, targets_path, pdb, pdb_name)
            except Exception as e:
                print(f"Skipping {pdb} as the TM-score calculation failed")
                tmscore, gdts = np.nan, np.nan

            try:
                lddt = get_lddt(f"{trafl_path}/{pdb}", f"{targets_path}/{pdb_name}", pdb, lddt_logs)
            except Exception as e:
                print(f"Skipping {pdb} as the lDDT calculation failed")
                lddt = np.nan
            outs.append((pdb, rms, round(ermsd, 3), round(inf, 3), tmscore, gdts, lddt))

        else:
            print(f"Skipping {pdb} as the target pdb file: {targets_path}/{pdb_name} does not exist")
    return outs

def align_biopython(trafl_path, targets_path, pdb, pdb_name):
    parser = PDBParser()
    ref_structure = parser.get_structure('ref', f"{targets_path}/{pdb_name}")
    ref_model = ref_structure[0]
    ref_atoms = [atom for atom in ref_model.get_atoms()]
    
    pred_structure = parser.get_structure('structure', f"{trafl_path}/{pdb}")
    model = pred_structure[0]
    atoms = [atom for atom in model.get_atoms()]
    sup = Superimposer()
    try:
        sup.set_atoms(ref_atoms, atoms) # if there is a missing P atom in some chain, then is should be removed from prediction to do superposition
        sup.apply(model.get_atoms())
        
    except Exception as e:
        print(f"Skipping {pdb} as the superimposition failed")
        
    return round(sup.rms, 3)
    
def align_pymol(trafl_path, targets_path, pdb, pdb_name):
    cmd.reinitialize()
    cmd.load(f"{targets_path}/{pdb_name}", "ref")
    cmd.load(f"{trafl_path}/{pdb}", "structure")
    rms = cmd.align("ref", "structure")[0]
    cmd.delete("ref")
    cmd.delete("structure")
    return round(rms, 3)

def main():
    args = parse_args()
    # out_postfix = '.seq-000001_AA.pdb'
    if args.sim_rna is not None:
        generate_pdbs_from_trafl(args.preds_path, args.templates_path, args.sim_rna, args.overwrite) #, out_postfix=out_postfix, pdb_postfix=".pdb")
    elif args.arena_path is not None:
        generate_pdbs_AA(args.preds_path, args.arena_path, args.overwrite, out_postfix='_AA.pdb')
    print("Superimposing...")
    outs = superimpose_pdbs(args.preds_path, args.targets_path , out_postfix="_AA.pdb", method="pymol", lddt_logs=args.lddt_logs)
    print("Results:")
    df = pd.DataFrame(outs, columns=['pdb', 'rms', 'ermsd', 'inf', 'tmscore', 'gdts', 'lddt'])
    # sort df by rms column
    df = df.sort_values(by='rms', ascending=True)
    print(df.head())
    print(f"Mean RMSD: {df['rms'].mean()}, Std: {df['rms'].std()}, Median RMSD: {df['rms'].median()}")
    print(f"Mean eRMSD: {df['ermsd'].mean()}, Std: {df['ermsd'].std()}, Median eRMSD: {df['ermsd'].median()}")
    print(f"Mean INF: {df['inf'].mean()}, Std: {df['inf'].std()}, Median INF: {df['inf'].median()}")
    print(f"Mean TM-score: {df['tmscore'].mean()}, Std: {df['tmscore'].std()}, Median TM-score: {df['tmscore'].median()}")
    print(f"Mean GDT-TS: {df['gdts'].mean()}, Std: {df['gdts'].std()}, Median GDT-TS: {df['gdts'].median()}")
    print(f"Mean lDDT: {df['lddt'].mean()}, Std: {df['lddt'].std()}, Median lDDT: {df['lddt'].median()}")
    print(f"Number of samples: {len(df)}")
    df.to_csv(args.output_name, index=False)
    print(f"Results saved to {args.output_name}")
    pass

if __name__ == "__main__":
    main()
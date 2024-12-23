import os
import numpy as np
from tqdm import tqdm
from rdkit import Chem
import pickle
import Bio
from Bio.PDB import PDBParser, MMCIFParser
from rnapolis.annotator import extract_secondary_structure
from rnapolis.parser import read_3d_structure
# from torch_geometric.data import Data
import warnings
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)
from constants import RESIDUES, ATOM_TYPES, RESIDUE_CONNECTION_GRAPH,\
    DOT_OPENINGS, DOT_CLOSINGS_MAP, KEEP_ELEMENTS, COARSE_GRAIN_MAP, ATOM_ELEMENTS

def load_molecule(molecule_file):
    if ".mol2" in molecule_file:
        my_mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=True)
    elif ".sdf" in molecule_file:
        suppl = Chem.SDMolSupplier(str(molecule_file), sanitize=False, removeHs=True)
        my_mol = suppl[0]
    elif ".pdb" in molecule_file:
        my_mol = Chem.MolFromPDBFile(
            str(molecule_file), sanitize=False, removeHs=True)
    else:
        raise ValueError("Unrecognized file type for %s" % str(molecule_file))
    if my_mol is None:
        raise ValueError("Unable to read non None Molecule Object")
    xyz = get_xyz_from_mol(my_mol)
    return xyz, my_mol

def load_with_bio(molecule_file, seq_segments, file_type:str=".pdb"):
    if file_type.endswith("pdb"):
        parser = PDBParser()
        structure = parser.get_structure("rna", molecule_file)
    else:
        parser = MMCIFParser()
        structure = parser.get_structure("rna", molecule_file)
    # generate full structure with all atoms
    coords, atoms_elements, atoms_names, residues_names, p_missing, c4_prime, c2, c4_or_c6, n1_or_n9, res_in_chain, coords_updated = generate_atoms(seq_segments)
    coords_in_residue = np.array(coords) # atoms order in coords: P, C4', Nx, C2, Cx
    coords_in_residue = coords_in_residue.reshape((-1, 5, 3)) # 5 atoms in each residue
    coords_updated = coords_updated.reshape((-1, 5))

    res_id = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                # HETATM are residues as well, skip them.
                if residue.id[0].startswith('H_') or\
                   residue.get_resname() not in RESIDUES.keys() or\
                   res_id >= len(seq_segments[0]): # sometimes there is 
                    continue
                
                if residue.get_resname() == 'U' or residue.get_resname() == 'C':
                    pur_pir_C = 'C4'
                    pur_pir_N = 'N1'
                elif residue.get_resname() == 'A' or residue.get_resname() == 'G':
                    pur_pir_C = 'C6'
                    pur_pir_N = 'N9'

                res_coords = coords_in_residue[res_id]
                res_coords_update = [False]*5
                for atom in residue:
                    if atom.get_name() == "P":
                        res_coords[0] = atom.get_coord()
                        res_coords_update[0] = True
                    elif atom.get_name() == "C4'":
                        res_coords[1] = atom.get_coord()
                        res_coords_update[1] = True
                    elif atom.get_name() == pur_pir_N:
                        res_coords[2] = atom.get_coord()
                        res_coords_update[2] = True
                    elif atom.get_name() == "C2":
                        res_coords[3] = atom.get_coord()
                        res_coords_update[3] = True
                    elif atom.get_name() == pur_pir_C:
                        res_coords[4] = atom.get_coord()
                        res_coords_update[4] = True
                if sum(res_coords_update) >= 3: # if at least 3 atoms are present, consider the residue as present
                    coords_in_residue[res_id] = res_coords
                    coords_updated[res_id] = res_coords_update
                    res_id += 1

    return coords_in_residue.reshape((-1, 3)), atoms_elements, atoms_names, residues_names, p_missing, c4_prime, c2, c4_or_c6, n1_or_n9, res_in_chain, coords_updated.reshape(-1)

def generate_atoms(seq_segments):
    coords = []
    atoms_elements = []
    atoms_names = []
    residues_names = []
    p_missing = []
    c4_prime = []
    c2 = []
    c4_or_c6 = []
    n1_or_n9 = []
    res_in_chain = []
    for segment in seq_segments:
        chain = 'A'
        for resi in segment:
            if resi == 'T': # in case of DNA sequences convert to RNA
                resi = 'U'
            for atom in COARSE_GRAIN_MAP[resi]:
                coords.append([0.,0.,0.])
                atoms_elements.append(ATOM_ELEMENTS[atom])
                atoms_names.append(atom)
                residues_names.append(resi)
                p_missing.append(False)
                c4_prime.append(atom == "C4'")
                c2.append(atom == "C2")
                c4_or_c6.append(atom == "C4" or atom == "C6")
                n1_or_n9.append(atom == "N1" or atom == "N9")
                res_in_chain.append(chain)
        chain = chr(ord(chain) + 1)
    coords_updated = [False]*len(coords)
    return np.array(coords), atoms_elements, atoms_names, residues_names, p_missing, c4_prime, c2, c4_or_c6, n1_or_n9, res_in_chain, np.array(coords_updated)

def get_xyz_from_mol(mol):
    xyz = np.zeros((mol.GetNumAtoms(), 3))
    conf = mol.GetConformer()
    for i in range(conf.GetNumAtoms()):
        position = conf.GetAtomPosition(i)
        xyz[i, 0] = position.x
        xyz[i, 1] = position.y
        xyz[i, 2] = position.z
    return (xyz)

def get_coarse_grain_mask(symbols, residues):
    coarse_atoms = [COARSE_GRAIN_MAP[x] for x in residues]
    mask = [True if atom in coars_atoms else False for atom, coars_atoms in zip(symbols, coarse_atoms)]
    return np.array(mask)

def get_edges_in_COO(data:dict, seq_segments:list[str], p_missing:list[bool], bpseq: list[tuple[int, int]] = None):
    # Order of encoded atoms: "P", "C4'", "Nx", "C2", "Cx"
    edges = []
    edge_type = [] # True: covalent, False: other interaction
    if seq_segments is not None:
        segments_lengs = [len(x) for x in seq_segments]
        segments_lengs = np.cumsum(segments_lengs) # get the end index of each segment
        indicies = np.concatenate([np.array([0]), segments_lengs[:-1]]).astype(int)        
    else:
        segments_lengs = []
        indicies = np.array([0])

    p = data['atoms'] == ATOM_TYPES['P']
    c4_prime = data['c4_primes']
    c2 = data['c2']
    c4_or_c6 = data['c4_or_c6']
    n1_or_n9 = data['n1_or_n9']
    nodes_indecies = np.arange(data['atoms'].shape[0])
    combined = np.stack([p, c4_prime, n1_or_n9, c2, c4_or_c6], axis=1)

    added = 0
    
    for index in indicies:
        if p_missing[index]: # the missing P can occur only in the first residue of the segment
            combined = np.concatenate([combined[:index*5], np.array([[True, False, False, False, False]]), combined[index*5:]])
            nodes_indecies = np.concatenate([nodes_indecies[:index*5], np.array([nodes_indecies[index*5]]), nodes_indecies[index*5:]]) # add "fake" P atom, with the same node index (that will be filtered out later).
            added += 1
    
    combined = combined.reshape((-1, 5, 5))
    nodes_indecies = nodes_indecies.reshape((-1, 5))
    comb_arg_max = np.argmax(combined, axis=2) # sometimes the order of atoms is 0,1,2,3,4, and sometimes it's different
    for res_ni, res_arg_max in zip(nodes_indecies, comb_arg_max): # create edges in each residue
        for i, j in RESIDUE_CONNECTION_GRAPH:
            edge = [res_ni[np.where(res_arg_max == i)[0]], res_ni[np.where(res_arg_max == j)[0]]]
            if edge[0] == edge[1]: # remove self loops, effect of adding missing P atoms
                continue
            edges.append(edge)
            edge_type.append(True)

    # connect residues
    for i in range(1, len(nodes_indecies)):
        if i in segments_lengs:
            continue
        prev_c4p = nodes_indecies[i-1][np.where(comb_arg_max[i-1] == 1)[0]] # C4' atom index in previous residue
        curr_p = nodes_indecies[i][np.where(comb_arg_max[i] == 0)[0]] # P atom index in current residue
        edges.append([prev_c4p, curr_p])
        edges.append([curr_p, prev_c4p])
        edge_type.extend([True, True]) # True means covalent bonds/backbone atoms

    # edges based on bpseq (2D structure)
    if bpseq is not None:
        for pair in bpseq:
            for i in range(2, 5): # atoms: N, C2, Cx
                at1 = nodes_indecies[pair[0]][np.where(comb_arg_max[pair[0]] == i)[0]] # atom i (e.g. N) connect with the corresponding atom in the paired residue
                at2 = nodes_indecies[pair[1]][np.where(comb_arg_max[pair[1]] == i)[0]]
                edges.append([at1, at2])
                edges.append([at2, at1])
                edge_type.extend([False, False]) # False - other interactions
    assert len(edges) == len(edge_type)
    return edges, edge_type

def read_seq_segments(seq_file):
    with open(seq_file, "r") as f:
        seq = f.readline()
    return seq.strip().split()

def bpseq_to_res_ids(bpseq):
    bpseq = bpseq.split("\n")
    bpseq = [x.split() for x in bpseq]
    bpseq = [(int(x[0])-1, int(x[2])-1) for x in bpseq if int(x[2]) != 0 and int(x[0]) < int(x[2])] # -1, because the indices in bpseq are 1-based, and we need 0-based (numpy indicies)
    return bpseq

def get_bpseq_pairs(rna_file, seq_path, extended_dotbracket=True):
    """
    If dotbracket file in seq_path is available, then read it and parse it to bpseq.
    Else Read 2D structure from 3D file.
    """
    if seq_path is not None:
        dot_file = seq_path.replace(".seq", ".dot")
        seq_segments = read_seq_segments(seq_path)
    else:
        dot_file = None
    if dot_file is not None and os.path.exists(dot_file):
        with open(dot_file) as f:
            dot = f.readlines() # the last line is dotbracket
    else:
        with open(rna_file) as f:
            structure3d = read_3d_structure(f, 1)
            structure2d = extract_secondary_structure(structure3d, 1)
        if extended_dotbracket: # include non-canonical pairings
            dot = structure2d.extendedDotBracket.split('\n')
        else:
            dot = structure2d.dotBracket.split('\n')
        dot, seq_segments = dotbrackets_to_single_line(dot)
    res_pairs = dot_to_bpseq(dot)
    return res_pairs, seq_segments

def dot_to_segments(dot):
    segments = [seg for seg in dot[1::3]]
    return segments

def dotbrackets_to_single_line(dot):
    segments = dot_to_segments(dot)
    dotb = [db for db in dot[2::3]]
    return dotb, segments

def dot_to_bpseq(dot):
    stack = {}
    bpseq = []
    dot_line = "".join(dot)
    for i, x in enumerate(dot_line):
        assert x in DOT_OPENINGS + list(DOT_CLOSINGS_MAP.keys()) + ["."], f"Invalid character in dotbracket: {x}"
        if x not in stack and x != ".":
                stack[x] = []
        if x in DOT_OPENINGS:
            stack[x].append(i)
        elif x in DOT_CLOSINGS_MAP:
            bpseq.append((stack[DOT_CLOSINGS_MAP[x]].pop(), i))
    return bpseq

def construct_graphs(seq_dir, pdbs_dir, save_dir, save_name, file_3d_type:str=".pdb", extended_dotbracket:bool=True, sampling:bool=False):
    """
    
    Args:
        seq_dir: directory with .seq files
        pdbs_dir: directory with 3D structures
        save_dir: directory to save the graphs
        save_name: name of the file to save the graphs
        file_3d_type: type of 3D structure files
        extended_dotbracket: if True, include non-canonical pairings in 2D structure
        sampling: if True, skips reading coordinates and generates fake atoms. For sampling ONLY.
    """
    save_dir_full = os.path.join(save_dir, save_name)

    if not os.path.exists(save_dir_full):
        os.makedirs(save_dir_full)
       
    if seq_dir is not None:
        name_list = [x for x in os.listdir(seq_dir)]
        name_list = [x for x in name_list if ".seq" in x]
    else:
        name_list = [x for x in os.listdir(pdbs_dir)]
        name_list = [x for x in name_list if file_3d_type in x]

    for i in tqdm(range(len(name_list))):
        name = name_list[i]
        
        
        if seq_dir is not None: # To remove
            seq_path = os.path.join(seq_dir, name)
            seq_segments = read_seq_segments(seq_path)
            name = name.replace(".seq", file_3d_type)
        else:
            seq_path = None
            seq_segments = None
        
        rna_file = os.path.join(pdbs_dir, name)
        
        # if rna_file exists, skip
        if os.path.exists(os.path.join(save_dir_full, name.replace(file_3d_type, ".pkl"))):
            continue
        if not os.path.exists(rna_file):
            print("File not found", rna_file)
            continue
        

        res_pairs, seq_segments = get_bpseq_pairs(rna_file, seq_path=seq_path, extended_dotbracket=extended_dotbracket)
        if not seq_segments:
            print("Error reading sequence", rna_file)
            continue


        if sampling:
            rna_coords, elements, atoms_symbols, residues_names, p_missing, c4_primes, c2, c4_or_c6, n1_or_n9, chains, coords_updated = generate_atoms(seq_segments)
        else:
            try:
                rna_coords, elements, atoms_symbols, residues_names, p_missing, c4_primes, c2, c4_or_c6, n1_or_n9, chains, coords_updated = load_with_bio(rna_file, seq_segments, file_3d_type)
            # except ValueError:
            #     print("Error reading molecule", rna_file)
            #     continue
            except Bio.PDB.PDBExceptions.PDBConstructionException as e:
                print("Error reading molecule (invalid or missing coordinate)", rna_file)
                continue

        elem_indices = set([i for i,x in enumerate(elements) if x in KEEP_ELEMENTS]) # keep only C, N, O, P atoms, remove all the others
        res_indices = set([i for i,x in enumerate(residues_names) if x in RESIDUES.keys()]) # keep only A, G, U, C residues, remove all the others
        x_indices = list(elem_indices.intersection(res_indices))
        elements = [elements[i] for i in x_indices]
        atoms_symbols = [atoms_symbols[i] for i in x_indices]
        residues_names = [residues_names[i] for i in x_indices]
        c4_primes = [c4_primes[i] for i in x_indices]
        c2 = [c2[i] for i in x_indices]
        c4_or_c6 = [c4_or_c6[i] for i in x_indices]
        n1_or_n9 = [n1_or_n9[i] for i in x_indices]
        rna_pos = np.array(rna_coords[x_indices])

        rna_x = np.array([ATOM_TYPES[x] for x in elements]) # Convert atomic numbers to types
        residues_x = np.array([RESIDUES[x] for x in residues_names]) # Convert residues to types

        assert len(rna_x) == len(rna_pos) == len(atoms_symbols) == len(residues_x) == len(c4_primes)
        if len(rna_pos) == 0:
            print("Structure contains too few atoms (e.g. backbone only).", rna_file)
            continue

        crs_gr_mask = get_coarse_grain_mask(atoms_symbols, residues_names)

        data = {}
        data['atoms'] = rna_x[crs_gr_mask]
        data['pos'] = rna_pos[crs_gr_mask]
        data['symbols'] = np.array(atoms_symbols)[crs_gr_mask]
        # data['indicator'] = graph_indicator[crs_gr_mask]
        data['name'] = name
        data['residues'] = residues_x[crs_gr_mask]
        data['c4_primes'] = np.array(c4_primes)[crs_gr_mask]
        data['c2'] = np.array(c2)[crs_gr_mask]
        data['c4_or_c6'] = np.array(c4_or_c6)[crs_gr_mask]
        data['n1_or_n9'] = np.array(n1_or_n9)[crs_gr_mask]
        data['chains'] = np.array(chains)[crs_gr_mask]
        data['coords_updated'] = np.array(coords_updated)[crs_gr_mask]
        try:
            edges, edge_type = get_edges_in_COO(data, seq_segments, p_missing=p_missing, bpseq=res_pairs)
        # except ValueError as e:
        #     print(f"Value Error in processing {name}: {e}")
        #     continue
        except IndexError as e:
            print(f"Index Error in processing {name}: {e}")
            continue
        data['edges'] = np.array(edges)
        data['edge_type'] = edge_type

        with open(os.path.join(save_dir_full, name.replace(file_3d_type, ".pkl")), "wb") as f:
            pickle.dump(data, f)


def main():
    extended_dotbracket = False
    data_dir = "/home/mjustyna/data/"
    # seq_dir = os.path.join(data_dir, "hl_seqs")
    # pdbs_dir = os.path.join(data_dir, "hl_pdbs")
    # save_dir = os.path.join(".", "data", "RNA-bgsu-hl-cn")
    # # construct_graphs(seq_dir, pdbs_dir, save_dir, "train-pkl", extended_dotbracket=extended_dotbracket)
    # construct_graphs(seq_dir, pdbs_dir, save_dir, "test-pkl", extended_dotbracket=extended_dotbracket)

    # data_dir = "/home/mjustyna/data/test_structs/"
    # seq_dir = os.path.join(data_dir, "seqs")
    # pdbs_dir = os.path.join(data_dir, "pdbs")

    data_dir = "/home/mjustyna/data//"
    seq_dir = None
    pdbs_dir = os.path.join(data_dir, "rna-solo")
    save_dir = os.path.join(".", "data", "rna-solo")
    construct_graphs(seq_dir, pdbs_dir, save_dir, "train-pkl", file_3d_type='.pdb', extended_dotbracket=extended_dotbracket, sampling=False)
    
    # data_dir = "/home/mjustyna/data/"
    # seq_dir = os.path.join(data_dir, "sim_desc")
    # pdbs_dir = os.path.join(data_dir, "rRNA_tRNA") #"desc-pdbs"
    
    

    # construct_graphs(seq_dir, pdbs_dir, save_dir, "rRNA_tRNA-train", extended_dotbracket=extended_dotbracket)
    # pdbs_dir = os.path.join(data_dir, "non_rRNA_tRNA")
    # construct_graphs(seq_dir, pdbs_dir, save_dir, "rRNA_tRNA-test", extended_dotbracket=extended_dotbracket)
    

if __name__ == "__main__":
    main()
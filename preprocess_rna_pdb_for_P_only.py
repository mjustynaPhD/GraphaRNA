import os
import numpy as np
from tqdm import tqdm
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

def load_with_bio(molecule_file, seq_segments, file_type:str=".pdb"):
    if file_type.endswith("pdb"):
        parser = PDBParser()
        structure = parser.get_structure("rna", molecule_file)
    else:
        parser = MMCIFParser()
        structure = parser.get_structure("rna", molecule_file)
    # generate full structure with all atoms
    coords, atoms_elements, atoms_names, residues_names, res_in_chain, coords_updated = generate_atoms(seq_segments)
    coords_in_residue = np.array(coords) # atoms order in coords: P
    coords_in_residue = coords_in_residue.reshape((-1, 3))

    res_id = 0
    for model in structure:
        for chain in model:
            for i, residue in enumerate(chain):
                # HETATM are residues as well, skip them.
                if residue.id[0].startswith('H_') or\
                   residue.get_resname() not in RESIDUES.keys() or\
                   res_id >= len("".join(seq_segments)): # sometimes there is 
                    continue
            
                for atom in residue:
                    if atom.get_name() == "P":
                        coords_in_residue[res_id] = atom.get_coord()
                        coords_updated[res_id] = True
                res_id+=1

    return coords_in_residue.reshape((-1, 3)), atoms_elements, atoms_names, residues_names, res_in_chain, coords_updated.reshape(-1)

def generate_atoms(seq_segments):
    coords = []
    atoms_elements = []
    atoms_names = []
    residues_names = []
    p_missing = []
    res_in_chain = []
    for segment in seq_segments:
        chain = 'A'
        for resi in segment:
            if resi == 'T': # in case of DNA sequences convert to RNA
                resi = 'U'
            atom = 'P'
            coords.append([0.,0.,0.])
            atoms_elements.append(ATOM_ELEMENTS[atom])
            atoms_names.append(atom)
            residues_names.append(resi)
            p_missing.append(False)
            res_in_chain.append(chain)
        chain = chr(ord(chain) + 1)
    coords_updated = [False]*len(coords)
    return np.array(coords), atoms_elements, atoms_names, residues_names, res_in_chain, np.array(coords_updated)

def get_coarse_grain_mask(symbols, residues):
    coarse_atoms = [COARSE_GRAIN_MAP[x] for x in residues]
    mask = [True if atom in coars_atoms else False for atom, coars_atoms in zip(symbols, coarse_atoms)]
    return np.array(mask)

def get_edges_in_COO(data:dict, seq_segments:list[str], bpseq: list[tuple[int, int]] = None):
    # Order of encoded atoms: "P", "C4'", "Nx", "C2", "Cx"
    edges = []
    edge_type = [] # True: covalent, False: other interaction
    if seq_segments is not None:
        segments_lengs = [len(x) for x in seq_segments]
        segments_lengs = np.cumsum(segments_lengs) # get the end index of each segment
    else:
        segments_lengs = []
    
    nodes_indecies = np.arange(data['atoms'].shape[0])

    # connect residues
    for i in range(1, len(nodes_indecies)):
        if i in segments_lengs: # split chains
            continue
        prev_p = nodes_indecies[i-1]
        curr_p = nodes_indecies[i]
        edges.append([prev_p, curr_p])
        edges.append([curr_p, prev_p])
        edge_type.extend([True, True]) # True means covalent bonds/backbone atoms

    # edges based on bpseq (2D structure)
    if bpseq is not None:
        for pair in bpseq:
            at1 = nodes_indecies[pair[0]]
            at2 = nodes_indecies[pair[1]]
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
        

        try:
            res_pairs, seq_segments = get_bpseq_pairs(rna_file, seq_path=seq_path, extended_dotbracket=extended_dotbracket)
        except IndexError:
            print("Error reading structure", rna_file)
            continue
        
        if not seq_segments:
            print("Error reading sequence", rna_file)
            continue


        if sampling:
            rna_coords, elements, atoms_symbols, residues_names, chains, coords_updated = generate_atoms(seq_segments)
        else:
            try:
                rna_coords, elements, atoms_symbols, residues_names, chains, coords_updated = load_with_bio(rna_file, seq_segments, file_3d_type)
            # except ValueError:
            #     print("Error reading molecule", rna_file)
            #     continue
            except Bio.PDB.PDBExceptions.PDBConstructionException:
                print("Error reading molecule (invalid or missing coordinate)", rna_file)

                continue

        elem_indices = set([i for i,x in enumerate(elements) if x in KEEP_ELEMENTS]) # keep only C, N, O, P atoms, remove all the others
        res_indices = set([i for i,x in enumerate(residues_names) if x in RESIDUES.keys()]) # keep only A, G, U, C residues, remove all the others
        x_indices = list(elem_indices.intersection(res_indices))
        elements = [elements[i] for i in x_indices]
        atoms_symbols = [atoms_symbols[i] for i in x_indices]
        residues_names = [residues_names[i] for i in x_indices]
        rna_pos = np.array(rna_coords[x_indices])

        rna_x = np.array([ATOM_TYPES[x] for x in elements]) # Convert atomic numbers to types
        residues_x = np.array([RESIDUES[x] for x in residues_names]) # Convert residues to types

        assert len(rna_x) == len(rna_pos) == len(atoms_symbols) == len(residues_x)
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
        data['chains'] = np.array(chains)[crs_gr_mask]
        data['coords_updated'] = np.array(coords_updated)[crs_gr_mask]
        try:
            edges, edge_type = get_edges_in_COO(data, seq_segments, bpseq=res_pairs)
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
    
    seq_dir = None
    pdbs_dir = os.path.join(data_dir, "full_PDB")
    save_dir = os.path.join(".", "data", "full_PDB-P")
    construct_graphs(seq_dir, pdbs_dir, save_dir, "train-pkl", file_3d_type='.pdb', extended_dotbracket=extended_dotbracket, sampling=False)
    

if __name__ == "__main__":
    main()
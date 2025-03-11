import os
import Bio
import Bio.PDB
import numpy as np
from torch import Tensor
from grapharna.constants import REV_ATOM_TYPES, REV_RESIDUES


class SampleToPDB():
    def __init__(self):
        pass

    def to(self, format: str, sample, path, name, post_fix:str=''):
        # Convert sample to the desired format
        unique_batches = np.unique(sample.batch.cpu().numpy())
        assert format in ['pdb', 'xyz', 'trafl'], f"Invalid format: {format}. Accepted formats: 'pdb', 'xyz', 'trafl'"
        for batch in unique_batches:
            mask = sample.batch == batch
            try:
                if format == 'pdb':
                    self.write_pdb(sample.x[mask], path, name[batch])
                elif format == 'xyz':
                    self.write_xyz(sample.x[mask], path, name[batch], post_fix)
                elif format == 'trafl':
                    self.write_trafl(sample.x[mask], path, name[batch])
            except ValueError as e:
                print("Cannot save molecules with missing P atom.")

    def write_xyz(self, x, path, name, post_fix:str='', rnd_dig:int=4):
        atoms = self.get_atoms_pos_and_types(x)
        atoms_pos = atoms['atoms_pos']
        atom_names = atoms['atom_names']

        name = name.replace(".pdb", "")
        name = name + post_fix
        name = name + '.xyz' if not name.endswith('.xyz') else name
        # Save the structure as a xyz file
        os.makedirs(path, exist_ok=True)
        out_path = os.path.join(path, name)
        with open(out_path, 'w') as f:
            f.write(f"{len(atoms_pos)}\n")
            f.write(f"{name}\n")
            for atom, pos in zip(atom_names, atoms_pos):
                f.write(f"{atom} {round(pos[0], rnd_dig)} {round(pos[1], rnd_dig)} {round(pos[2], rnd_dig)}\n")

    def write_trafl(self, x: Tensor, path:str, name:str, post_fix:str='', rnd_dig:int=4):
        """The trafl format was described in SimRNA Manual. Here is the quote:
        "The format of coordinates line is just:
        x1 y1 z1 x2 y2 z2 … xN yN zN
        The coordinates of the subsequent points corresponds to the following order of the atoms: P, C4’,
        N(N1 or N9 for pyrimidine or purine, respectively), B1 (C2), B2 (C4 or C6 for pyrimidine or purine,
        respectively). In general, the coordinate line will contain 5*numberOfNucleotides points, so
        3*5*numberOfNucleotides coordinate items (in 3D space: 3 coordinates per atom, 5 atoms per
        residue; hence 15 coordinates per residue)." - SimRNA Manual

        Args:
            x (Tensor): input tensor of coordinates and atom types
            path (_type_): Path were output file should be saved
            name (_type_): Name of the output file
            post_fix (str, optional): Postfix added to the name of file (if any). Defaults to ''.
            rnd_dig (int, optional): Round coordinates to n decimal points. Defaults to 4.
        Raises:
            ValueError: If the number of atoms is not divisible by 5. The number of atoms should be a multiple of 5. Cannot save to trafl with missing P atom.
        """
        atoms = self.get_atoms_pos_and_types(x)
        atoms_pos = atoms['atoms_pos']
        atom_names = atoms['atom_names']
        atom_names = np.array(atom_names)
        if len(atoms_pos) % 5 != 0:
            raise ValueError("The number of atoms is not divisible by 5. The number of atoms should be a multiple of 5. Cannot save to trafl with missing P atom.")
        atoms_pos = atoms_pos.reshape(-1, 5, 3)
        atom_names = atom_names.reshape(-1, 5)
        p_atom = x[:, -9].cpu().numpy()
        p_c4p_c2_c46_n19 = x[:, -4:].cpu().numpy()
        p_c4p_c2_c46_n19 = np.concatenate([p_atom.reshape(-1, 1), p_c4p_c2_c46_n19], axis=1)
        p_c4p_c2_c46_n19 = p_c4p_c2_c46_n19.reshape(-1, 5, 5)

        name = name.replace(".pdb", "")
        name = name + post_fix
        name = name + '.trafl' if not name.endswith('.trafl') else name
        # Save the structure as a trafl file
        os.makedirs(path, exist_ok=True)
        out_path = os.path.join(path, name)
        c=0
        with open(out_path, 'w') as f:
            header = "1 1 0 0 0"
            f.write(header + "\n")
            for atom, pos, orders in zip(atom_names, atoms_pos, p_c4p_c2_c46_n19):
                argmaxs = np.argmax(orders, axis=1)
                save_order = np.array([
                    np.where(argmaxs == 0)[0], # P
                    np.where(argmaxs == 1)[0], # C4'
                    np.where(argmaxs == 4)[0], # N1 or N9
                    np.where(argmaxs == 2)[0], # C2
                    np.where(argmaxs == 3)[0] # C4 or C6
                ]).flatten()
                for atom_name, atom_pos in zip(atom[save_order], pos[save_order]):
                    f.write(f" {atom_pos[0]:.3f} {atom_pos[1]:.3f} {atom_pos[2]:.3f}")
                    c+=1
        pass

    def write_pdb(self, x, path, name):
        name = name.replace(".pdb", "").replace(".cif", "")
        atoms = self.get_atoms_pos_and_types(x)
    
        structure = self.create_structure(atoms, name)
        
        name = name + '.pdb' if not name.endswith('.pdb') else name
        # Save the structure as a PDB file
        os.makedirs(path, exist_ok=True)
        out_path = os.path.join(path, name)
        io = Bio.PDB.PDBIO()
        io.set_structure(structure)
        io.save(out_path)

    def get_atoms_pos_and_types(self, x):
        atoms_pos = x[:, :3].cpu().numpy()
        # atoms_pos *= 10
        atoms_types = x[:, 3:7].cpu().numpy()
        atom_names = [REV_ATOM_TYPES[np.argmax(atom)] for atom in atoms_types]
        residues = x[:, 7:11].cpu().numpy()
        residues = np.argmax(residues, axis=1)
        c4_prime = x[:, 11].cpu().numpy()
        c2 = x[:, 12].cpu().numpy()
        c4_or_c6 = x[:, 13].cpu().numpy()
        n1_or_n9 = x[:, 14].cpu().numpy()
        out = {
            'atoms_pos': atoms_pos,
            'atom_names': atom_names,
            'residues': residues,
            'c4_prime': c4_prime,
            'c2': c2,
            'c4_or_c6': c4_or_c6,
            'n1_or_n9': n1_or_n9
        }
        return out


    def create_structure(self, atoms, name):
        # Create an empty structure
        structure = Bio.PDB.Structure.Structure(name)
        
        # Create a model within the structure
        model = Bio.PDB.Model.Model(0)
        structure.add(model)
        
        # Create a chain within the model
        chain = Bio.PDB.Chain.Chain('A')
        model.add(chain)
        
        coords = atoms['atoms_pos']
        atoms_names = atoms['atom_names']
        residues = atoms['residues']
        c4_primes = atoms['c4_prime']
        c2 = atoms['c2']
        c4_or_c6 = atoms['c4_or_c6']
        n1_or_n9 = atoms['n1_or_n9']

        # Create atoms and add them to the chain
        residue_id = 0
        atoms_added = 0
        assert len(atoms_names) % 5 == 0, "The number of atoms should be a multiple of 5"
        
        for coord, atom, res, c4p, c2, c4or6, n1or9 in zip(coords, atoms_names, residues, c4_primes, c2, c4_or_c6, n1_or_n9):
            residue_name = REV_RESIDUES[res]
            if atoms_added % 5 == 0:
                residue_id += 1
                residue = Bio.PDB.Residue.Residue((' ', residue_id, ' '), residue_name, ' ')
                chain.add(residue)
            
            if residue_name == 'A' or residue_name == 'G':
                c46_name = 'C6'
                n19_name = 'N9'
            elif residue_name == 'U' or residue_name == 'C':
                c46_name = 'C4'
                n19_name = 'N1'

            # Create an atom
            if atom == 'P':
                new_atom = Bio.PDB.Atom.Atom(atom, coord, 0, 0, ' ', atom, 0, atom)
            elif c4p:
                new_atom = Bio.PDB.Atom.Atom('C4\'', coord, 0, 0, ' ', 'C4\'', 0, atom)
            elif c2:
                new_atom = Bio.PDB.Atom.Atom('C2', coord, 0, 0, ' ', 'C2', 0, atom)
            elif c4or6:
                new_atom = Bio.PDB.Atom.Atom(c46_name, coord, 0, 0, ' ', c46_name, 0, atom)
            elif n1or9:
                new_atom = Bio.PDB.Atom.Atom(n19_name, coord, 0, 0, ' ', n19_name, 0, atom)

            residue.add(new_atom)
            atoms_added += 1
            
        return structure
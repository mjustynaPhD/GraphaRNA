import numpy as np
import Bio.PDB as PDB


def kabsch_numpy(A, B):
    """
    Computes the optimal rotation (R) and translation (t)
    to align A (Nx3) onto B (Nx3).
    """
    # Compute centroids
    C_A = np.mean(A, axis=0)
    C_B = np.mean(B, axis=0)

    # Center the point clouds
    A_centered = A - C_A
    B_centered = B - C_B

    # Compute correlation matrix
    H = A_centered.T @ B_centered

    # SVD decomposition
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation
    R = Vt.T @ U.T

    # Ensure right-handed coordinate system
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
    
    R = Vt.T @ U.T

    # Compute translation
    t = C_B - R @ C_A

    return R, t

def load_pdb_file(f):
    """
    Load a PDB file and return the coordinates of the alpha carbons.

    :param f: The path to the PDB file.
    :return: A Nx3 matrix of alpha carbon coordinates.
    """
    parser = PDB.PDBParser()
    structure = parser.get_structure('structure', f)
    model = structure[0]
    return model

def rotate_models(model1, model2):
    ref_residue = model1['A'][2]
    # coords of P, C4' and N1
    P_ref = ref_residue['P'].get_vector()
    C4_ref = ref_residue['C4\''].get_vector()
    N_ref = ref_residue['N1'].get_vector()
    # convert vectors to numpy arrays
    P_ref = np.array([P_ref[0], P_ref[1], P_ref[2]])
    C4_ref = np.array([C4_ref[0], C4_ref[1], C4_ref[2]])
    N_ref = np.array([N_ref[0], N_ref[1], N_ref[2]])
    # print residue name and number
    print(ref_residue.get_resname(), ref_residue.get_id()[1])

    for chain in model2:
        for residue in chain:
            # if residue contains P, C4' and N1 or N9 atoms
            if 'P' in residue and 'C4\'' in residue and 'N1' in residue:
                P = residue['P'].get_vector()
                C4 = residue['C4\''].get_vector()
                N = residue['N1'].get_vector()
                # convert vectors to numpy arrays
                P = np.array([P[0], P[1], P[2]])
                C4 = np.array([C4[0], C4[1], C4[2]])
                N = np.array([N[0], N[1], N[2]])
                # print residue name and number
                print(residue.get_resname(), residue.get_id()[1])
                # Compute the rotation matrix and translation vector
                R, t = kabsch_numpy(np.array([P, C4, N]),
                                    np.array([P_ref, C4_ref, N_ref]))
                break

    # transform model2
    for chain in model2:
        for residue in chain:
            for atom in residue:
                atm_vec = np.array([i for i in atom.get_vector()])
                atom.set_coord(R @ atm_vec + t)
    return model1, model2

def main():
    # Load the PDB files
    path = 'equivariance_tests'
    model1 = load_pdb_file(f'{path}/1ehz.pdb')
    model2 = load_pdb_file(f'{path}/1ffk_9.pdb')
    rot1, rot2 = rotate_models(model1, model2)
    # save rotated model2
    io = PDB.PDBIO()
    io.set_structure(rot2)
    io.save(f'{path}/rotated.pdb')

if __name__ == '__main__':
    main()
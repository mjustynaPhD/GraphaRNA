# implement gram-schmidt process

def gram_schmidt(vectors):
    # input: list of vectors
    # output: list of orthogonal vectors
    basis = []
    for v in vectors:
        w = v - sum(np.dot(v, b) * b for b in basis)
        if (w > 1e-10).any():
            basis.append(w / np.linalg.norm(w))
    return basis
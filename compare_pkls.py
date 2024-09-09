import os
import pickle
import numpy as np

D1 = "data/RNA-PDB/desc-pkl/"
D2 = "data/RNA-PDB/desc-pkl-v2/"

def main():
    f1 = sorted(os.listdir(D1))
    f2 = sorted(os.listdir(D2))
    for a, b in zip(f1, f2):
        assert a == b
        with open(D1 + a, 'rb') as f:
            d1 = pickle.load(f)
        with open(D2 + b, 'rb') as f:
            d2 = pickle.load(f)
        e1 = np.array(d1['edges'])
        e2 = np.squeeze(d2['edges'], axis=2)
        assert e1 == e2
    pass

if __name__ == "__main__":
    main()

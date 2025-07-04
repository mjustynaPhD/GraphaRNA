import os
from pathlib import Path

preds_dir = Path("/home/mjustyna/RNA-GNN/samples/grapharna-eval-seed=0/800/")
out_dir = Path("/home/mjustyna/RNA-GNN/samples/grapharna-eval-seed=0/")

with open("all-output.txt", 'r') as f:
    lines = f.readlines()

lines = [l.split() for l in lines]
lines = [l for l in lines if len(l)>=7 and len(l[1])>=12]
pairs = dict([(l[1], int(l[2])) for l in lines])
seqs = dict([(l[1], l[-pairs[l[1]]:]) for l in lines])
len_seqs = dict([(k, len("".join(v))) for k, v in seqs.items()])

for output in preds_dir.iterdir():
    basename = output.name
    if not basename.endswith("_AA.pdb"):
        continue
    print(f"Processing {basename}")
    name = basename.split("_AA.pdb")[0]
    if name not in pairs:
        print(f"Skipping {name}, not in pairs")
        continue
    
    num_segments = pairs[name]
    #copy the file
    save_dir = out_dir / f"{num_segments}_segments"
    save_dir.mkdir(parents=True, exist_ok=True)
    source = preds_dir / basename
    target = save_dir / basename
    if not target.exists():
        print(f"Copying {source} to {target}")
        target.write_bytes(source.read_bytes())
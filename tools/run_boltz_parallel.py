import sys
import os

FASTA_DIR = "all_3seg_fastas"

def main(current_index:int, all_jobs:int):
    files = sorted(os.listdir(FASTA_DIR))
    batch_len = len(files) // int(all_jobs)
    start = int(current_index) * batch_len
    end = start + batch_len
    job_files = files[start:end]
    for j in job_files:
        fasta_path = os.path.join(FASTA_DIR, j)
        cmd = f"boltz predict {fasta_path}"
        print(f"Running command: {cmd}")
        os.system(cmd)


if __name__ == "__main__":
    current_index, all_jobs = sys.argv[1], sys.argv[2]
    main(current_index, all_jobs)

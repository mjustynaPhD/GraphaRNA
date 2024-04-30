#!/bin/bash
file1=$1
file2=$2

cd /home/macieka/2d-analysis/
sudo docker-compose run --rm --name `cat /proc/sys/kernel/random/uuid` --entrypoint ./descs descs --execution-mode DESCRIPTORS_COMPARISON --file-path-of-first-descriptor $file1 --file-path-of-second-descriptor $file2 --molecule-type rna -cat BACKTRACKING_DRIVEN_FIRST_ALIGNMENT_ONLY --maximal-rmsd-of-central-elements-alignment 2.5 --maximal-rmsd-of-pair-of-aligned-duplexes 4.0 -maep 0.51 -aan /data/3d/desc-test/in-contact-residues-identification-based-on-c5prim-only.exp --output-directory /data/3d/desc-test/out
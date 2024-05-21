#!/bin/bash

P="/home/mjustyna/data/sim_desc"
OUT="/home/mjustyna/data/sim_pdb"

for f in $P/*;
do

	echo $f
	base_name="$(basename $f)"
	echo $base_name
	./SimRNA -s $f
	mv $base_name*pdb $OUT/
	rm $base_name*	

done

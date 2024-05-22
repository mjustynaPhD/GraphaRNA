#!/bin/bash

awk -v elemsratio=0.8 -v resratio=0.66 -f promising-descs-pairs.awk /data/3d/input_data/descs/segments_2-output.txt > descs-seg-2.csv
cd /data/3d/input_data/descs/
cat *-output.txt > all-output.txt
cat all-output.txt | awk 'BEGIN {FS="\t";} NF==7' > filter-output.txt
cat filter-output.txt | awk 'BEGIN {FS="\t";} $3==3'
cat filter-output.txt | awk 'BEGIN {FS="\t";} $3==3 && $5<=60' > segments_3-output.txt
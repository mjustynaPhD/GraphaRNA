cd /data/3d/input_data/descs/
# cat *-output.txt > all-output.txt
# cat all-output.txt | awk 'BEGIN {FS="\t";} NF==7' > filter-output.txt
cat filter-output.txt | awk 'BEGIN {FS="\t";} $3==2' # deskryptory 2 segmentowe
cat filter-output.txt | awk 'BEGIN {FS="\t";} $3==2 && $5<=60' > segments_2-output.txt
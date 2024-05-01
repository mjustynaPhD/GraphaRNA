function min(x,y)
{
    if (x < y)
        return x
    else
        return y
}
function max(x,y)
{
    if (x > y)
        return x
    else
        return y
}
BEGIN {FS="\t";}
NF==7 {
	id=$2;
	segno[id]=$3;
	elemno[id]=$4;
	resno[id]=$5;
	d[++i]=id;
}
END	{
	for(j=1;j<=i;j++) {
		id1 = d[j];
		for(k=j+1;k<=i;k++) {
			id2 = d[k];
			if ((segno[id1] == segno[id2]) && (min(elemno[id1],elemno[id2])/max(elemno[id1],elemno[id2])>=elemsratio) && (min(resno[id1],resno[id2])/max(resno[id1],resno[id2])>=resratio))
				print id1 "," id2;
		}
	}
}
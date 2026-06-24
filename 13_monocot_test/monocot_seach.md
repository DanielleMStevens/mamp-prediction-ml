
# Combine homology search and mamp-ml to rapid identification and characterization of receptor variants

First, we will download both the maize B73 and wheat Kronos genome from NCBI and XX, respecitly. [Link for B73](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000005005.2/)

```
# load blast module on cluster
bio/blast-plus/2.14.1-gcc-11.4.0

# blastp search v6 genome
blastp -query riceFLS2.fasta -subject GCA_000005005.6_B73_RefGen_v4_protein.faa -out maize_FLS2_hits.txt -outfmt 6

# check hits and make sure coverage seems reasonable
head -n 6 maize_FLS2_hits.txt | awk '{print $1, $2, $3, $11, $12}' | column -t

OsFLS2  ONM13841.1  69.237  0.0        1496
OsFLS2  AQL09482.1  34.872  0.0        575
OsFLS2  ONM57413.1  34.005  7.56e-179  561
OsFLS2  AQK82441.1  33.333  1.59e-176  551
OsFLS2  ONM51377.1  33.656  3.95e-174  544
OsFLS2  ONM16079.1  32.084  1.08e-173  543

# pull cds sequences 
zcat GCA_000005005.6_B73_RefGen_v4_cds_from_genomic.fna.gz | awk 'NR==FNR {if(FNR<=6 && $2!="") ids[$2]; next} /^>/ {p=0; for(id in ids) if($0 ~ id) {p=1; break}} p' maize_FLS2_hits.txt - > top_6_maize_dna_cds.fasta


```
```
# for searching the wheat genome
blastp -query riceFLS2.fasta -subject Kronos_v2_protein.fasta -out wheat_FLS2_hits.txt -outfmt 6

# check hits and make sure coverage seems reasonable
head -n 6 wheat_FLS2_hits.txt | awk '{print $1, $2, $3, $11, $12}' | column -t

OsFLS2  TrturKRN2A02G065640.2  72.938  0.0       1610
OsFLS2  TrturKRN2B02G075960.3  72.251  0.0       1570
OsFLS2  TrturKRN2B02G075960.1  69.218  0.0       1501
OsFLS2  TrturKRN2A02G065640.1  55.546  0.0       1126
OsFLS2  TrturKRN2B02G075960.2  69.171  0.0       989
OsFLS2  TrturKRN2B02G075960.2  75.449  1.36e-68  250

# pull cds sequences
zcat Kronos_v2_CDS.fasta.gz | awk 'NR==FNR {if(FNR<=6 && $2!="") ids[$2]; next} /^>/ {p=0; for(id in ids) if($0 ~ id) {p=1; break}} p' wheat_FLS2_hits.txt - > top_6_wheat_dna_cds.fasta


```

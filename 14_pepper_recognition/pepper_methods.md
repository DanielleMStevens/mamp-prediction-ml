
Search for pepper genomes
select those with contig/chromosome number less than 1000 -> 25 genomes
copy genbank accessions

```
conda create -n ncbi_datasets

conda activate ncbi_datasets
conda install -c conda-forge ncbi-datasets-cli

# loop through accessions
while read -r acc; do datasets download genome accession "$acc" --include gbff --dehydrated --filename "${acc}.zip"; done < pepper_accessions.txt


# unzip all accessions:
while read -r acc; do unzip "${acc}.zip" -d "${acc}"; datasets rehydrate --directory "${acc}"; done < pepper_accessions.txt

# rehydrate all genomes
for file in *.zip; do dir="${file%.zip}"; unzip "$file" -d "$dir"; datasets rehydrate --directory "$dir"; done
```


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

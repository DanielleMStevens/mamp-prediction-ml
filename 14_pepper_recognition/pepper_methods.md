
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

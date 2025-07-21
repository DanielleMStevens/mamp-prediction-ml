# NCBI Datasets

## remove small proteins (less than 250 in length)
conda install -c bioconda seqkit
seqkit seq -m 250 -g VITVvi_vCabSauv08_v1.fasta > VITVvi_vCabSauv08_v1_filtered.fasta

## download tmhmm 
conda install -c bioconda phobius



# Search all the protein fasta files for the kinase doamin
download hmm model for kinase domain: https://www.ebi.ac.uk/interpro/entry/pfam/PF00069/logo/
conda install -c bioconda hmmer
hmmsearch -A kinase_alignment.stk --tblout kinase_domains.txt -E 1 --domE 1 --incE 0.01 --incdomE 0.04 --cpu 8 PF00069.hmm VITVvi_vCabSauv08_v1_filtered.fasta 

# Convert the output from hmmersearch into a fasta file
esl-reformat fasta kinase_alignment.stk > reformat_kinase_hits.fasta
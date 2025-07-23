#!/usr/bin/env Rscript

# Script to extract full-length sequences based on kinase hits
# Usage: This script reads reformat_kinase_hits.fasta to get gene IDs,
# then extracts corresponding full-length sequences from VITVvi_vCabSauv08_v1_filtered.fasta

library(Biostrings)

# Define file paths
kinase_hits_file <- "11_grape_analysis/reformat_kinase_hits.fasta"
full_length_file <- "11_grape_analysis/VITVvi_vCabSauv08_v1_filtered.fasta"
output_file <- "11_grape_analysis/full_length_kinase_hits.fasta"

# Function to extract gene ID from header
extract_gene_id <- function(header) {
  # Extract the part before the first "/"
  gene_id <- gsub("/.*$", "", header)
  return(gene_id)
}

# Read kinase hits file and extract unique gene IDs
kinase_hits <- readAAStringSet(kinase_hits_file)
kinase_headers <- names(kinase_hits)

# Extract unique gene IDs
unique_gene_ids <- unique(sapply(kinase_headers, extract_gene_id))

# Debug: show first few examples
cat("Example kinase hit header:", kinase_headers[1], "\n")
cat("Extracted gene ID:", unique_gene_ids[1], "\n")

# Read full-length sequences
full_length_seqs <- readAAStringSet(full_length_file)
full_length_headers <- names(full_length_seqs)

# Debug: show first few full-length headers
cat("Example full-length header:", full_length_headers[1], "\n")

# Find matches and extract sequences
matching_indices <- which(full_length_headers %in% unique_gene_ids)

if (length(matching_indices) > 0) {
  # Extract matching sequences
  extracted_seqs <- full_length_seqs[matching_indices]
  
  # Write to output file
  writeXStringSet(extracted_seqs, output_file)
  
  # Print summary
  cat("Extracted", length(extracted_seqs), "full-length sequences from", length(unique_gene_ids), "unique gene IDs\n")
  cat("Output written to:", output_file, "\n")
  
} else {
  cat("No matching sequences found! Check that gene IDs match between files.\n")
  cat("First few unique gene IDs from kinase hits:\n")
  print(head(unique_gene_ids))
  cat("First few headers from full-length file:\n") 
  print(head(full_length_headers))
}
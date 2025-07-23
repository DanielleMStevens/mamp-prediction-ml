#!/usr/bin/env Rscript

# Script to extract full-length sequences based on LRR domain hits
# Usage: This script reads combine_lrr_hmmer_hits.fasta to get gene IDs,
# then extracts corresponding full-length sequences from filtered_tm_sp_proteins.fasta

library(Biostrings)

# Input and output files for LRR domain analysis
lrr_hits_file <- "11_grape_analysis/combine_lrr_hmmer_hits.fasta"
full_length_file <- "11_grape_analysis/filtered_tm_sp_proteins.fasta"
output_file <- "11_grape_analysis/full_length_lrr_hits.fasta"

# Function to extract gene ID from header
extract_gene_id <- function(header) {
  # Extract the part before the first "/"
  gene_id <- gsub("/.*$", "", header)
  return(gene_id)
}

# Function to clean full-length headers by removing metadata
clean_full_length_header <- function(header) {
  # Remove everything from " |" onwards to get just the gene ID
  clean_id <- gsub(" \\|.*$", "", header)
  return(clean_id)
}

# Read LRR hits file and extract unique gene IDs
lrr_hits <- readAAStringSet(lrr_hits_file)
lrr_headers <- names(lrr_hits)

# Extract unique gene IDs (this handles redundancy where same protein appears multiple times)
unique_gene_ids <- unique(sapply(lrr_headers, extract_gene_id))

# Debug: show first few examples
cat("Example LRR hit header:", lrr_headers[1], "\n")
cat("Extracted gene ID:", unique_gene_ids[1], "\n")

# Read full-length sequences
full_length_seqs <- readAAStringSet(full_length_file)
full_length_headers <- names(full_length_seqs)

# Clean full-length headers to remove metadata
clean_full_length_headers <- sapply(full_length_headers, clean_full_length_header)

# Debug: show first few full-length headers
cat("Example full-length header:", full_length_headers[1], "\n")
cat("Cleaned full-length header:", clean_full_length_headers[1], "\n")

# Find matches and extract sequences
matching_indices <- which(clean_full_length_headers %in% unique_gene_ids)

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
  cat("First few unique gene IDs from LRR hits:\n")
  print(head(unique_gene_ids))
  cat("First few cleaned headers from full-length file:\n") 
  print(head(clean_full_length_headers))
}
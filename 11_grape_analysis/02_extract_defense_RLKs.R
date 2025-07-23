# Improved script with better debugging and duplicate handling

# Read clade file and clean gene names
clade_lines <- readLines("vitus_RLK_defense_clade.txt")
# Remove empty lines and process
clade_lines <- clade_lines[clade_lines != ""]
clade_genes <- gsub(" ", "_", sub("/.*", "", clade_lines))

# Remove duplicates and show counts
unique_clade_genes <- unique(clade_genes)
cat("Total lines in clade file:", length(clade_lines), "\n")
cat("Unique genes in clade file:", length(unique_clade_genes), "\n")
cat("Duplicates removed:", length(clade_genes) - length(unique_clade_genes), "\n")

# Read FASTA file
fasta_lines <- readLines("filtered_tm_sp_proteins.fasta")
header_lines <- grep("^>", fasta_lines)

# Get all FASTA gene names
fasta_gene_names <- c()
for (i in seq_along(header_lines)) {
  header <- fasta_lines[header_lines[i]]
  gene_name <- sub(" .*", "", sub("^>", "", header))
  fasta_gene_names <- c(fasta_gene_names, gene_name)
}

# Find matches
found_genes <- unique_clade_genes[unique_clade_genes %in% fasta_gene_names]
missing_genes <- unique_clade_genes[!unique_clade_genes %in% fasta_gene_names]

cat("Genes found in FASTA:", length(found_genes), "\n")
cat("Genes NOT found in FASTA:", length(missing_genes), "\n")

# Show missing genes if any
if (length(missing_genes) > 0) {
  cat("\nMissing genes:\n")
  for (i in 1:min(10, length(missing_genes))) {
    cat(missing_genes[i], "\n")
  }
  if (length(missing_genes) > 10) cat("... and", length(missing_genes) - 10, "more\n")
}

# Extract sequences and check for duplicates
output_lines <- c()
unique_seqs <- list()  # Store unique sequences

for (i in seq_along(header_lines)) {
  header <- fasta_lines[header_lines[i]]
  gene_name <- sub(" .*", "", sub("^>", "", header))
  
  if (gene_name %in% unique_clade_genes) {
    # Get sequence lines
    start <- header_lines[i]
    end <- ifelse(i < length(header_lines), header_lines[i+1] - 1, length(fasta_lines))
    seq_lines <- fasta_lines[(start+1):end]
    seq <- paste(seq_lines, collapse="")  # Full sequence
    
    # Only add if sequence is unique
    if (!seq %in% names(unique_seqs)) {
      unique_seqs[[seq]] <- header
      output_lines <- c(output_lines, header, seq_lines)
    }
  }
}

# Write output
writeLines(output_lines, "vitus_RLK_defense_clade_sequences.fasta")
cat("\nDone! Wrote", length(grep("^>", output_lines)), "sequences (after removing duplicates) to output file\n")
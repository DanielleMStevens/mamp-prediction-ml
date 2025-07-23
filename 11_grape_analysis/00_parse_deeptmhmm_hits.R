# Simple R script to filter DeepTMHMM predictions and create FASTA file
# Only keeps entries with TM, SP, or SP+TM predictions (excludes GLOB)

# Input and output file paths
input_file <- "11_grape_analysis/combined_deeptmhmm_predictions.txt"
output_file <- "11_grape_analysis/filtered_tm_sp_proteins.fasta"

# Read all lines from the input file
cat("Reading input file...\n")
lines <- readLines(input_file)

# Initialize vectors to store filtered headers and sequences
filtered_headers <- c()
filtered_sequences <- c()

# Process the file (every 3 lines: header, sequence, topology)
cat("Processing entries...\n")
i <- 1
total_entries <- 0
kept_entries <- 0

while (i <= length(lines)) {
  # Check if this is a header line
  if (startsWith(lines[i], ">")) {
    total_entries <- total_entries + 1
    
    # Extract header and check prediction type
    header <- lines[i]
    
    # Check if it contains TM, SP, or SP+TM (but not GLOB)
    if (grepl("\\| (TM|SP|SP\\+TM)$", header) && !grepl("\\| GLOB$", header)) {
      # This entry should be kept
      sequence <- lines[i + 1]  # Next line is the sequence
      
      # Add to our filtered lists
      filtered_headers <- c(filtered_headers, header)
      filtered_sequences <- c(filtered_sequences, sequence)
      kept_entries <- kept_entries + 1
    }
    
    # Skip to next entry (3 lines per entry)
    i <- i + 3
  } else {
    i <- i + 1
  }
}

# Write the filtered FASTA file
cat("Writing output file...\n")
output_lines <- c()
for (j in 1:length(filtered_headers)) {
  output_lines <- c(output_lines, filtered_headers[j], filtered_sequences[j])
}

writeLines(output_lines, output_file)

# Print summary
cat("Processing complete!\n")
cat("Total entries processed:", total_entries, "\n")
cat("Entries kept (TM/SP/SP+TM):", kept_entries, "\n")
cat("Entries excluded (GLOB):", total_entries - kept_entries, "\n")
cat("Output written to:", output_file, "\n")
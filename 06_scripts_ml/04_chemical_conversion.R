# Load required packages
library(tidyverse)
library(Peptides)

# Read the training data
train_data <- read.csv("datasets/train_stratify.csv")
test_data <- read.csv("datasets/test_stratify.csv")

# Function to convert sequence to bulkiness values
sequence_to_bulkiness <- function(sequence) {
  # Split sequence into individual amino acids
  amino_acids <- strsplit(sequence, "")[[1]]
  
  # Create a named vector of bulkiness values
  bulkiness_data <- c(
    A = 11.50, R = 14.28, N = 12.82, D = 11.68, C = 13.46,
    Q = 14.45, E = 13.57, G = 3.40, H = 13.69, I = 21.40,
    L = 21.40, K = 15.71, M = 16.25, F = 19.80, P = 17.43,
    S = 9.47, T = 15.77, W = 21.67, Y = 18.03, V = 21.57
  )
  
  # Convert each amino acid to its bulkiness value
  bulkiness_values <- sapply(amino_acids, function(aa) {
    if (aa %in% names(bulkiness_data)) {
      return(bulkiness_data[aa])
    } else {
      return(NA)  # Return NA for invalid/unknown amino acids
    }
  })
  
  # Convert to comma-separated string
  return(paste(bulkiness_values, collapse = ","))
}

# Convert both Sequence and Receptor.Sequence to bulkiness values
train_data$Sequence_Bulkiness <- sapply(train_data$Sequence, sequence_to_bulkiness)
train_data$Receptor_Bulkiness <- sapply(train_data$Receptor.Sequence, sequence_to_bulkiness)

test_data$Sequence_Bulkiness <- sapply(test_data$Sequence, sequence_to_bulkiness)
test_data$Receptor_Bulkiness <- sapply(test_data$Receptor.Sequence, sequence_to_bulkiness)

# update colnames
colnames_chemical_names <-  c("Plant species","Receptor","Locus ID/Genbank","Epitope",
"Sequence","Known Outcome","Receptor Name","Receptor Sequence","Sequence_Bulkiness","Receptor_Bulkiness")
colnames(train_data) <- colnames_chemical_names
colnames(test_data) <- colnames_chemical_names

# Save the processed data
write.csv(train_data, "datasets/processed/train_data_with_bulkiness.csv", row.names = FALSE)
write.csv(test_data, "datasets/processed/test_data_with_bulkiness.csv", row.names = FALSE)


# Print first few entries to verify
#print("First few rows of processed data:")
#print(head(train_data[c("Sequence", "Sequence_Bulkiness", "Receptor.Sequence", "Receptor_Bulkiness")])) 
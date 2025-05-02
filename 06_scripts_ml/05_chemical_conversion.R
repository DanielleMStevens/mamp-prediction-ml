# Load required packages
library(tidyverse)
library(Peptides)


#args <- commandArgs(trailingOnly = TRUE)
#if (length(args) == 0) {
#  stop("Usage: Rscript 06_scripts_ml/04_chemical_conversion.R <feature1> [feature2] ...\n",
#       "   or: Rscript 06_scripts_ml/04_chemical_conversion.R all\n",
 #      "   Example: Rscript 06_scripts_ml/04_chemical_conversion.R bulkiness charge\n",
 #      "   Available features: bulkiness, charge, hydrophobicity, all")
#}

# --- Define Available Features ---
#selected_features <- args
#all_available_features <- c("bulkiness", "charge", "hydrophobicity")

# --- Process Selected Features ---
# If "all" is specified (as the only argument), use all available features.
# Case-insensitive check for "all"
#if (length(selected_features) == 1 && tolower(selected_features[1]) == "all") {#
#  selected_features <- all_available_features
#} else {
  # Validate that selected features are among the available ones (case-insensitive)
#  selected_features_lower <- tolower(selected_features)
# invalid_features <- selected_features[!selected_features_lower %in% all_available_features]

#  if (length(invalid_features) > 0) {
#    stop("Invalid feature(s) specified: ", paste(invalid_features, collapse=", "),
#         ".\nChoose from: ", paste(all_available_features, collapse=", "), " or 'all'.")
#  }
   # Ensure we use the canonical names (lowercase) for processing if needed
#  selected_features <- selected_features_lower
#}
# Ensure uniqueness (although parsing multiple args handles this naturally)
#selected_features <- unique(selected_features)


# Read the training data
train_data <- read.csv("05_datasets/train_stratify.csv")
test_data <- read.csv("05_datasets/test_stratify.csv")

# Function to convert sequence to bulkiness values
sequence_to_bulkiness <- function(sequence) {
  # Split sequence into individual amino acids
  amino_acids <- strsplit(sequence, "")[[1]]
  
  # Create a named vector of bulkiness values
  bulkiness_data <- c(
  A = 11.50, R = 14.28, N = 12.82, D = 11.68, C = 13.46, Q = 14.45, E = 13.57,
  G = 3.40, H = 13.69, I = 21.40, L = 21.40, K = 15.71, M = 16.25, F = 19.80,
  P = 17.43, S = 9.47, T = 15.77, W = 21.67, Y = 18.03, V = 21.57
  )
  charge_data <- c(
    A = 0, R = 1, N = 0, D = -1, C = 0, Q = 0, E = -1, G = 0, H = 0.1,
    I = 0, L = 0, K = 1, M = 0, F = 0, P = 0, S = 0, T = 0, W = 0, Y = 0, V = 0
  )
  hydrophobicity_manavalan_data <- c(
    A = 0.61, R = 0.00, N = 0.06, D = 0.06, C = 1.07, Q = 0.00, E = 0.01,
    G = 0.74, H = 0.61, I = 2.22, L = 1.53, K = 0.28, M = 1.18, F = 2.02,
    P = 1.95, S = 0.46, T = 0.45, W = 2.65, Y = 1.88, V = 1.32
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
write.csv(train_data, "05_datasets/train_data_with_bulkiness.csv", row.names = FALSE)
write.csv(test_data, "05_datasets/test_data_with_bulkiness.csv", row.names = FALSE)


# Print first few entries to verify
#print("First few rows of processed data:")
#print(head(train_data[c("Sequence", "Sequence_Bulkiness", "Receptor.Sequence", "Receptor_Bulkiness")])) 
# Load required packages
library(tidyverse)
# library(Peptides) # Loaded but not used in this version, keep for potential future use

# --- Get Command Line Arguments ---
# Get all command line arguments after the script name
args <- commandArgs(trailingOnly = TRUE)

# --- Validate User Input ---
# Check if any arguments were provided
if (length(args) == 0) {
  stop("Usage: Rscript 06_scripts_ml/04_chemical_conversion.R <feature1> [feature2] ...\n",
       "   or: Rscript 06_scripts_ml/04_chemical_conversion.R all\n",
       "   Example: Rscript 06_scripts_ml/04_chemical_conversion.R bulkiness charge\n",
       "   Available features: bulkiness, charge, hydrophobicity, all")
}

# Assign the arguments to selected_features
selected_features <- args

# --- Define Available Features ---
all_available_features <- c("bulkiness", "charge", "hydrophobicity")

# --- Process Selected Features ---
# If "all" is specified (as the only argument), use all available features.
# Case-insensitive check for "all"
if (length(selected_features) == 1 && tolower(selected_features[1]) == "all") {
  selected_features <- all_available_features
} else {
  # Validate that selected features are among the available ones (case-insensitive)
  selected_features_lower <- tolower(selected_features)
  invalid_features <- selected_features[!selected_features_lower %in% all_available_features]

  if (length(invalid_features) > 0) {
    stop("Invalid feature(s) specified: ", paste(invalid_features, collapse=", "),
         ".\nChoose from: ", paste(all_available_features, collapse=", "), " or 'all'.")
  }
   # Ensure we use the canonical names (lowercase) for processing if needed
  selected_features <- selected_features_lower
}
# Ensure uniqueness (although parsing multiple args handles this naturally)
selected_features <- unique(selected_features)


# --- Read Data ---
# Assuming standard R data frame column names (dots instead of spaces).
# Add error handling for file reading
train_data <- tryCatch(
  read.csv("05_datasets/train_stratify.csv", check.names = FALSE), # keep original names
  error = function(e) stop("Error reading train_stratify.csv: ", e$message)
)
test_data <- tryCatch(
  read.csv("05_datasets/test_stratify.csv", check.names = FALSE), # keep original names
  error = function(e) stop("Error reading test_stratify.csv: ", e$message)
)

# Check if the required sequence columns exist
required_cols <- c("Sequence", "Receptor Sequence") # Use exact names expected from CSV
if (!all(required_cols %in% colnames(train_data))) {
  stop("Missing required columns in training data. Needed: '", paste(required_cols, collapse="', '"), "'. Found: '", paste(colnames(train_data), collapse="', '"), "'")
}
if (!all(required_cols %in% colnames(test_data))) {
    stop("Missing required columns in test data. Needed: '", paste(required_cols, collapse="', '"), "'. Found: '", paste(colnames(test_data), collapse="', '"), "'")
}


# --- Define Feature Scales ---
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

# --- Feature Conversion Function ---
sequence_to_feature_values <- function(sequence, feature = "bulkiness") {
  # Add debug print
  print(paste("Processing sequence:", sequence))
  
  # Return NA string representation if input is not a single character string
  if (!is.character(sequence) || length(sequence) != 1 || is.na(sequence)) {
      print("Invalid sequence detected")
      return(NA_character_)
  }
  
  # Return empty string if sequence is empty
  if (nchar(sequence) == 0) {
       print("Empty sequence detected")
       return("")
  }

  amino_acids <- strsplit(sequence, "")[[1]]
  
  feature_scale <- switch(feature,
    "bulkiness" = bulkiness_data,
    "charge" = charge_data,
    "hydrophobicity" = hydrophobicity_manavalan_data,
    stop(paste("Internal error: Invalid feature passed to conversion function:", feature))
  )

  # Get feature values, handling unknown amino acids
  feature_values <- sapply(amino_acids, function(aa) {
    val <- feature_scale[toupper(aa)]
    ifelse(is.null(val) || is.na(val), NA, val)
  }, USE.NAMES = FALSE)

  # Add debug print for feature values
  print(paste("Feature values:", paste(feature_values, collapse=",")))

  feature_values_formatted <- ifelse(is.na(feature_values), "NA", sprintf("%.2f", feature_values))
  return(paste(feature_values_formatted, collapse = ","))
}


# --- Process Data ---
# Store original column names before adding new ones
original_colnames_train <- colnames(train_data)
original_colnames_test <- colnames(test_data)

# Loop through selected features and add new columns
print(paste("Processing features:", paste(selected_features, collapse=", ")))
for (feature in selected_features) {
  # Generate descriptive column names (e.g., Sequence_Bulkiness)
  # Ensure first letter is capitalized
  feature_title_case <- paste0(toupper(substring(feature, 1, 1)), substring(feature, 2))
  seq_col_name <- paste0("Sequence_", feature_title_case)
  rec_col_name <- paste0("Receptor_", feature_title_case)

  # Apply the conversion function to create new columns
  # Use the exact column names read from the CSV
  train_data[[seq_col_name]] <- sapply(train_data[["Sequence"]], sequence_to_feature_values, feature = feature)
  train_data[[rec_col_name]] <- sapply(train_data[["Receptor.Sequence"]], sequence_to_feature_values, feature = feature)

  test_data[[seq_col_name]] <- sapply(test_data[["Sequence"]], sequence_to_feature_values, feature = feature)
  test_data[[rec_col_name]] <- sapply(test_data[["Receptor.Sequence"]], sequence_to_feature_values, feature = feature)

  print(paste("Added columns:", seq_col_name, "and", rec_col_name))
}

# --- Save Data ---
# Generate a suffix for filenames based on selected features
features_suffix <- paste(selected_features, collapse = "_and_")
train_output_file <- paste0("05_datasets/train_data_with_", features_suffix, ".csv")
test_output_file <- paste0("05_datasets/test_data_with_", features_suffix, ".csv")

# Write the updated data frames to new CSV files
write.csv(train_data, train_output_file, row.names = FALSE)
write.csv(test_data, test_output_file, row.names = FALSE)

print(paste("Processed training data saved to:", train_output_file))
print(paste("Processed test data saved to:", test_output_file))

# --- Verification ---
# Print the head of the training data including original sequences and newly added feature columns
print("First few rows of processed training data (selected columns):")
# Identify newly added columns based on the difference in column count
num_new_cols = length(selected_features) * 2
# Handle case where train_data might be empty or have fewer columns than expected
if (ncol(train_data) >= num_new_cols) {
    new_col_indices <- (ncol(train_data) - num_new_cols + 1):ncol(train_data)
    cols_to_print <- c("Sequence", "Receptor.Sequence", colnames(train_data)[new_col_indices])
} else {
    # Fallback if something went wrong
    cols_to_print <- colnames(train_data)
    warning("Could not reliably determine newly added columns for verification print.")
}

# Use tryCatch for printing head in case columns don't exist as expected
tryCatch(
  print(head(train_data[, cols_to_print, drop = FALSE])),
  error = function(e) warning("Could not print head of data: ", e$message)
) 
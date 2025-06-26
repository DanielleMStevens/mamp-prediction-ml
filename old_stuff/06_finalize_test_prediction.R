#-----------------------------------------------------------------------------------------------
# Krasileva Lab - Plant & Microbial Biology Department UC Berkeley
# Author: Danielle M. Stevens
# Script Purpose: Merge test predictions from two CSV files
#-----------------------------------------------------------------------------------------------

library(tidyverse)

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 2) {
  stop("This script requires exactly 2 CSV file paths as arguments")
}

csv_file1 <- args[1]
csv_file2 <- args[2]

# Check if files exist
if (!file.exists(csv_file1)) {
  stop(paste("File not found:", csv_file1))
}
if (!file.exists(csv_file2)) {
  stop(paste("File not found:", csv_file2))
}

# Read the CSV files
test_data <- read.csv(csv_file1)
predictions <- read.csv(csv_file2)

# Combine the data
# Since the files should have matching rows in the same order, we can use bind_cols
combined_data <- bind_cols(
  test_data,
  predictions %>% select(prob_class0, prob_class1, prob_class2, predicted_label)
)
# reorder columns
combined_data <- combined_data[c(1,2,3,4,5,6,9,10,11,12,7,8)]

# Map the numeric labels to meaningful categories
combined_data <- combined_data %>%
  mutate(
    predicted_label = case_when(
      predicted_label == 0 ~ "Immunogenic",
      predicted_label == 1 ~ "Non-Immunogenic", 
      predicted_label == 2 ~ "Weakly Immunogenic",
      TRUE ~ "Unknown"
    )
)


# Write the combined data to a new CSV file
output_file <- "combined_test_predictions.csv"
write.csv(combined_data, output_file, row.names = FALSE)

cat("Combined data has been written to:", output_file, "\n")



#!/usr/bin/env Rscript

#-----------------------------------------------------------------------------------------------
# Script Purpose: Create t-SNE visualization from MAMP ML model predictions
# Author: Assistant  
# Created for: Brian Su
# Description: This script creates a t-SNE plot using the prediction probabilities as 
#              a proxy for embeddings, and provides a framework for full embedding extraction.
#-----------------------------------------------------------------------------------------------

# Load required libraries
library(dplyr)
library(ggplot2)
library(Rtsne)
library(readr)
library(viridis)
library(patchwork)

# Function to create t-SNE from prediction probabilities
#create_tsne_from_predictions <- function(predictions_file, output_dir = ".") {
  
  cat("Loading prediction data...\n")
  
  # Load test predictions
  pred_data <- read_csv(file.choose())
  
  cat("Loaded", nrow(pred_data), "predictions\n")
  
  # Use prediction probabilities as feature representation
  prob_features <- pred_data %>%
    select(prob_class0, prob_class1, prob_class2) %>%
    as.matrix()
  
  # Add some derived features for better separation
  prob_features_enhanced <- cbind(
    prob_features,
    max_prob = apply(prob_features, 1, max),
    entropy = -rowSums(prob_features * log(prob_features + 1e-10)),
    prob_diff_01 = abs(prob_features[,1] - prob_features[,2]),
    prob_diff_02 = abs(prob_features[,1] - prob_features[,3]),
    prob_diff_12 = abs(prob_features[,2] - prob_features[,3])
  )
  
  cat("Running t-SNE on probability features...\n")
  
  # Run t-SNE
  set.seed(42)
  perplexity <- min(30, floor((nrow(prob_features_enhanced) - 1) / 3))
  
  tsne_result <- Rtsne(prob_features_enhanced, 
                       dims = 2, 
                       perplexity = perplexity,
                       max_iter = 1000,
                       pca = TRUE,
                       normalize = TRUE,
                       verbose = TRUE)
  
  # Create plotting data frame
  tsne_df <- data.frame(
    tSNE1 = tsne_result$Y[, 1],
    tSNE2 = tsne_result$Y[, 2],
    true_label = factor(pred_data$true_label, 
                       levels = c(0, 1, 2), 
                       labels = c("Non-Immunogenic", "Immunogenic", "Weakly Immunogenic")),
    predicted_label = factor(pred_data$predicted_label, 
                            levels = c(0, 1, 2),
                            labels = c("Non-Immunogenic", "Immunogenic", "Weakly Immunogenic")),
    prob_class0 = pred_data$prob_class0,
    prob_class1 = pred_data$prob_class1,
    prob_class2 = pred_data$prob_class2,
    max_prob = apply(prob_features, 1, max),
    correct_prediction = pred_data$true_label == pred_data$predicted_label
  )
  
  # Create plots
  plots <- list()
  
  # Plot 1: Colored by true labels
  plots$true_labels <- ggplot(tsne_df, aes(x = tSNE1, y = tSNE2, color = true_label)) +
    geom_point(alpha = 0.8, size = 2.5) +
    scale_color_viridis_d(name = "True Label", option = "plasma") +
    theme_minimal() +
    theme(
      panel.grid = element_blank(),
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      legend.position = "bottom",
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 12)
    ) +
    labs(
      title = "t-SNE of MAMP ML Model Predictions",
      subtitle = "Colored by True Immunogenicity Labels",
      x = "t-SNE Dimension 1",
      y = "t-SNE Dimension 2"
    ) +
    guides(color = guide_legend(override.aes = list(size = 4)))
  
  # Plot 2: Colored by predicted labels
  plots$predicted_labels <- ggplot(tsne_df, aes(x = tSNE1, y = tSNE2, color = predicted_label)) +
    geom_point(alpha = 0.8, size = 2.5) +
    scale_color_viridis_d(name = "Predicted Label", option = "plasma") +
    theme_minimal() +
    theme(
      panel.grid = element_blank(),
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      legend.position = "bottom",
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 12)
    ) +
    labs(
      title = "t-SNE of MAMP ML Model Predictions",
      subtitle = "Colored by Predicted Immunogenicity Labels",
      x = "t-SNE Dimension 1",
      y = "t-SNE Dimension 2"
    ) +
    guides(color = guide_legend(override.aes = list(size = 4)))
  
  # Plot 3: Colored by prediction accuracy
  plots$accuracy <- ggplot(tsne_df, aes(x = tSNE1, y = tSNE2, color = correct_prediction)) +
    geom_point(alpha = 0.8, size = 2.5) +
    scale_color_manual(values = c("FALSE" = "#E31A1C", "TRUE" = "#33A02C"),
                       name = "Prediction",
                       labels = c("Incorrect", "Correct")) +
    theme_minimal() +
    theme(
      panel.grid = element_blank(),
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      legend.position = "bottom",
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 12)
    ) +
    labs(
      title = "t-SNE of MAMP ML Model Predictions",
      subtitle = "Colored by Prediction Accuracy",
      x = "t-SNE Dimension 1",
      y = "t-SNE Dimension 2"
    ) +
    guides(color = guide_legend(override.aes = list(size = 4)))
  
  # Plot 4: Colored by prediction confidence (max probability)
  plots$confidence <- ggplot(tsne_df, aes(x = tSNE1, y = tSNE2, color = max_prob)) +
    geom_point(alpha = 0.8, size = 2.5) +
    scale_color_viridis_c(name = "Max\nProbability", option = "inferno") +
    theme_minimal() +
    theme(
      panel.grid = element_blank(),
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      legend.position = "bottom",
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 12)
    ) +
    labs(
      title = "t-SNE of MAMP ML Model Predictions",
      subtitle = "Colored by Prediction Confidence",
      x = "t-SNE Dimension 1",
      y = "t-SNE Dimension 2"
    )
  
  # Combined plot
  plots$combined <- (plots$true_labels + plots$predicted_labels) / 
                   (plots$accuracy + plots$confidence) +
    plot_annotation(
      title = "MAMP ML Model Prediction Analysis via t-SNE",
      subtitle = "Visualization based on prediction probabilities",
      theme = theme(plot.title = element_text(size = 16, face = "bold"))
    )
  
  # Save plots
  cat("Saving plots to", output_dir, "...\n")
  
  # Individual plots
  ggsave(filename = file.path(output_dir, "tsne_true_labels.pdf"),
         plot = plots$true_labels, width = 10, height = 8, dpi = 300)
  
  ggsave(filename = file.path(output_dir, "tsne_predicted_labels.pdf"),
         plot = plots$predicted_labels, width = 10, height = 8, dpi = 300)
  
  ggsave(filename = file.path(output_dir, "tsne_accuracy.pdf"),
         plot = plots$accuracy, width = 10, height = 8, dpi = 300)
  
  ggsave(filename = file.path(output_dir, "tsne_confidence.pdf"),
         plot = plots$confidence, width = 10, height = 8, dpi = 300)
  
  # Combined plot
  ggsave(filename = file.path(output_dir, "tsne_combined_analysis.pdf"),
         plot = plots$combined, width = 16, height = 12, dpi = 300)
  
  # Save data
  write_csv(tsne_df, file.path(output_dir, "tsne_coordinates.csv"))
  
  # Print summary statistics
  cat("\n=== Analysis Summary ===\n")
  cat("Total samples:", nrow(tsne_df), "\n")
  cat("Accuracy:", round(mean(tsne_df$correct_prediction) * 100, 1), "%\n")
  
  label_counts <- table(tsne_df$true_label)
  cat("\nTrue label distribution:\n")
  print(label_counts)
  
  pred_label_counts <- table(tsne_df$predicted_label)
  cat("\nPredicted label distribution:\n")
  print(pred_label_counts)
  
  cat("\nFiles saved:\n")
  cat("- tsne_true_labels.pdf\n")
  cat("- tsne_predicted_labels.pdf\n") 
  cat("- tsne_accuracy.pdf\n")
  cat("- tsne_confidence.pdf\n")
  cat("- tsne_combined_analysis.pdf\n")
  cat("- tsne_coordinates.csv\n")
  
  return(list(plots = plots, data = tsne_df))
}


# -------------------------\

# Load required Python modules
torch <- import("torch")
sys <- import("sys")
sys$path c(sys$path, "./06_scripts_ml")
  
models <- import("models.esm_positon_weighted")
datasets <- import("datasets.seq_with_receptor_dataset")
  
# Load model
checkpoint <- torch$load("/Users/briansu/workspace/mamp_prediction_ml/07_model_results/00_mamp_ml/checkpoint-19.pth", map_location = "cpu")
  
# Create model with proper args
args <- list(bfactor_csv_path = NULL)
model <- models$ESMBfactorWeightedFeatures(r_to_py(args), num_classes = 3L)
#model$load_state_dict(checkpoint$model_state_dict)
model$eval()
  
# Load test data with sequences
test_data <- read_csv(file = "/Users/briansu/workspace/mamp_prediction_ml/05_datasets/test_data_with_all_test_immuno_stratify.csv")
  
# Convert to PyTorch dataset
dataset <- datasets$PeptideSeqWithReceptorDataset(r_to_py(test_data))
  
# Create data loader
torch_utils <- import("torch.utils.data")
dataloader <- torch_utils$DataLoader(
    dataset, 
    batch_size = 32L, 
    shuffle = FALSE,
    collate_fn = model$collate_fn
)
  
  # Extract embeddings
  embeddings <- list()
  
  # Forward pass through model up to embedding layer
  # This requires modifying the forward pass to return embeddings
  # instead of final classification logits
  with(torch$no_grad(), {
    for (batch in reticulate::iter(dataloader)) {
      embed <- model$extract_embeddings(batch)
      if (!is.null(embed)) {
        numpy_embed <- embed$cpu()$numpy()
        embeddings[[length(embeddings) + 1]] <- numpy_embed
      }
    }
  })
  
  cat("MAMP ML Model Prediction Analysis with t-SNE\n")
  cat("============================================\n\n")
  
  # Define paths
  predictions_file <- "./07_model_results/00_mamp_ml/test_predictions.csv"
  output_dir <- "./07_model_results/00_mamp_ml/"
  
  # Check if files exist
  if (!file.exists(predictions_file)) {
    stop("Test predictions file not found at: ", predictions_file)
  }
  
  # Create output directory if it doesn't exist
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  # Run analysis
  result <- create_tsne_from_predictions(predictions_file, output_dir)
  
  cat("\n=== Next Steps for Full Embedding Extraction ===\n")
  cat("To extract actual model embeddings (not just prediction probabilities):\n")
  cat("1. Locate the original test dataset with peptide and receptor sequences\n")
  cat("2. Modify the model's forward() method to return embeddings before classification\n")
  cat("3. Use the template below for implementation\n\n")
  
  # Show template
  create_embedding_extraction_template()
  
  return(result)
}

# Run if script is executed directly
if (!interactive()) {
  result <- main()
}
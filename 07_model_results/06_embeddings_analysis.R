#!/usr/bin/env Rscript

#-----------------------------------------------------------------------------------------------
# Script Purpose: Extract embeddings from MAMP ML model and create t-SNE visualization
# Author: Assistant
# Created for: Brian Su
# Description: This script loads the trained MAMP prediction model, extracts embeddings 
#              from the test data, and creates a t-SNE plot for visualization.
#-----------------------------------------------------------------------------------------------

# Load required libraries
library(reticulate)
library(dplyr)
library(ggplot2)
library(Rtsne)
library(readr)
library(viridis)

# Set up Python environment (adjust path as needed)
# Uncomment and modify the line below if you need to specify a specific Python environment
use_python("/Users/briansu/anaconda3/envs/esmfold/bin/python")

# Import required Python libraries
# Try to import torch, with error handling
torch <- reticulate::import("torch")
pd <- reticulate::import("pandas")
np <- reticulate::import("numpy")
sys <- reticulate::import("sys")

# Add the model scripts directory to Python path
sys$path <- c(sys$path, "./06_scripts_ml")

# Import the model class
models <- import("models.esm_positon_weighted")
datasets <- import("datasets.seq_with_receptor_dataset") 

# Function to load model and extract embeddings
extract_embeddings <- function(model_path, data_path, device = "cpu") {
  
  cat("Loading model checkpoint...\n")
  
  # Import argparse first
  argparse <- reticulate::import("argparse")
  pathlib <- reticulate::import("pathlib")
  PosixPath <- pathlib$PosixPath
  
  # Add argparse.Namespace to safe globals before loading model
  torch$serialization$add_safe_globals(list(argparse$Namespace, PosixPath))
  
  # Load the checkpoint with weights_only=FALSE
  checkpoint <- torch$load(model_path, map_location = device, weights_only = FALSE)
  
  # Create model instance (you may need to adjust args based on your model)
  # Create a minimal args object with required parameters
  args <- list(
    model_checkpoint_path = NULL,
    bfactor_csv_path = NULL
  )
  
  # Convert to Python dict-like object
  args_py <- r_to_py(args)
  
  # Initialize model
  model <- models$ESMBfactorWeightedFeatures(args_py, num_classes = 3L)
  
  # Load state dict
  model$load_state_dict(checkpoint$model)
  model$eval()
  model$to(device)
  
  cat("Model loaded successfully!\n")
  
  # Load test data
  cat("Loading test data...\n")
  test_data <- read_csv(data_path)
  
  # Create dataset object
  test_dataset <- datasets$SeqWithReceptorDataset(
    sequences = test_data$sequence,
    receptors = test_data$receptor,
    labels = test_data$true_label,
    tokenizer_name = "facebook/esm2_t6_8M_UR50D"
  )
  
  # Extract embeddings
  cat("Extracting embeddings...\n")
  
  embeddings <- list()
  labels <- list()
  predicted_labels <- list()
  probabilities <- list()
  
  # Process batches
  batch_size <- 32
  for (i in seq(1, length(test_dataset), by = batch_size)) {
    end_idx <- min(i + batch_size - 1, length(test_dataset))
    batch_indices <- i:end_idx
    
    # Get batch data
    batch <- test_dataset[batch_indices]
    
    # Forward pass through model
    with(torch$no_grad(), {
      output <- model$forward_embeddings(
        batch$peptide_tokens$to(device),
        batch$receptor_tokens$to(device)
      )
      
      # Store embeddings and predictions
      embeddings[[length(embeddings) + 1]] <- output$embeddings$cpu()$numpy()
      labels[[length(labels) + 1]] <- batch$label$cpu()$numpy()
      predicted_labels[[length(predicted_labels) + 1]] <- output$predictions$argmax(dim = 2L)$cpu()$numpy()
      probabilities[[length(probabilities) + 1]] <- output$predictions$softmax(dim = 2L)$cpu()$numpy()
    })
  }
  
  # Combine batches
  embeddings <- do.call(rbind, embeddings)
  labels <- unlist(labels)
  predicted_labels <- unlist(predicted_labels)
  probabilities <- do.call(rbind, probabilities)
  
  return(list(
    embeddings = embeddings,
    labels = labels,
    predicted_labels = predicted_labels,
    probabilities = probabilities
  ))
}

# Function to create t-SNE plot
create_tsne_plot <- function(embeddings, labels, predicted_labels = NULL, 
                            perplexity = 30, max_iter = 1000) {
  
  cat("Running t-SNE...\n")
  
  # Run t-SNE
  set.seed(42)  # For reproducibility
  tsne_result <- Rtsne(embeddings, 
                       dims = 2, 
                       perplexity = perplexity,
                       max_iter = max_iter,
                       pca = TRUE,
                       normalize = FALSE,
                       verbose = TRUE)
  
  # Create data frame for plotting
  tsne_df <- data.frame(
    tSNE1 = tsne_result$Y[, 1],
    tSNE2 = tsne_result$Y[, 2],
    true_label = factor(labels, levels = c(0, 1, 2), 
                       labels = c("Non-Immunogenic", "Immunogenic", "Weakly Immunogenic"))
  )
  
  if (!is.null(predicted_labels)) {
    tsne_df$predicted_label <- factor(predicted_labels, levels = c(0, 1, 2),
                                     labels = c("Non-Immunogenic", "Immunogenic", "Weakly Immunogenic"))
    tsne_df$correct_prediction <- tsne_df$true_label == tsne_df$predicted_label
  }
  
  # Create plots
  plots <- list()
  
  # Plot colored by true labels
  plots$true_labels <- ggplot(tsne_df, aes(x = tSNE1, y = tSNE2, color = true_label)) +
    geom_point(alpha = 0.7, size = 2) +
    scale_color_viridis_d(name = "True Label") +
    theme_minimal() +
    theme(
      panel.grid = element_blank(),
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      legend.position = "bottom"
    ) +
    labs(
      title = "t-SNE Visualization of MAMP ML Model Embeddings",
      subtitle = "Colored by True Immunogenicity Labels",
      x = "t-SNE Dimension 1",
      y = "t-SNE Dimension 2"
    )
  
  if (!is.null(predicted_labels)) {
    # Plot colored by predicted labels
    plots$predicted_labels <- ggplot(tsne_df, aes(x = tSNE1, y = tSNE2, color = predicted_label)) +
      geom_point(alpha = 0.7, size = 2) +
      scale_color_viridis_d(name = "Predicted Label") +
      theme_minimal() +
      theme(
        panel.grid = element_blank(),
        axis.text = element_blank(),
        axis.ticks = element_blank(),
        legend.position = "bottom"
      ) +
      labs(
        title = "t-SNE Visualization of MAMP ML Model Embeddings",
        subtitle = "Colored by Predicted Immunogenicity Labels",
        x = "t-SNE Dimension 1",
        y = "t-SNE Dimension 2"
      )
    
    # Plot colored by correctness
    plots$prediction_accuracy <- ggplot(tsne_df, aes(x = tSNE1, y = tSNE2, color = correct_prediction)) +
      geom_point(alpha = 0.7, size = 2) +
      scale_color_manual(values = c("FALSE" = "#d62728", "TRUE" = "#2ca02c"),
                        name = "Prediction",
                        labels = c("Incorrect", "Correct")) +
      theme_minimal() +
      theme(
        panel.grid = element_blank(),
        axis.text = element_blank(),
        axis.ticks = element_blank(),
        legend.position = "bottom"
      ) +
      labs(
        title = "t-SNE Visualization of MAMP ML Model Embeddings",
        subtitle = "Colored by Prediction Accuracy",
        x = "t-SNE Dimension 1",
        y = "t-SNE Dimension 2"
      )
  }
  
  return(list(plots = plots, data = tsne_df))
}

# Main execution
main <- function() {
  cat("Starting MAMP ML embeddings extraction and t-SNE analysis...\n")
  
  # Define paths
  model_path <- "./07_model_results/00_mamp_ml/checkpoint-19.pth"
  data_path <- "./05_datasets/test_data_with_all_test_immuno_stratify.csv"
  output_dir <- "./tSNE_plots/"
  
  # Extract embeddings
  result <- extract_embeddings(model_path, data_path)
  
  # Create t-SNE visualization
  tsne_result <- create_tsne_plot(
    embeddings = result$embeddings,
    labels = result$labels,
    predicted_labels = result$predicted_labels,
    perplexity = min(30, floor((nrow(result$embeddings) - 1) / 3))  # Adjust perplexity if needed
  )
  
  # Save plots
  cat("Saving plots...\n")
  
  ggsave(filename = file.path(output_dir, "tsne_true_labels.pdf"),
         plot = tsne_result$plots$true_labels,
         width = 10, height = 8, dpi = 300)
  
  if ("predicted_labels" %in% names(tsne_result$plots)) {
    ggsave(filename = file.path(output_dir, "tsne_predicted_labels.pdf"),
           plot = tsne_result$plots$predicted_labels,
           width = 10, height = 8, dpi = 300)
    
    ggsave(filename = file.path(output_dir, "tsne_prediction_accuracy.pdf"),
           plot = tsne_result$plots$prediction_accuracy,
           width = 10, height = 8, dpi = 300)
  }
  
  # Save t-SNE coordinates
  write_csv(tsne_result$data, file.path(output_dir, "tsne_coordinates.csv"))
  
  cat("Analysis complete! Files saved to:", output_dir, "\n")
  cat("Generated files:\n")
  cat("- tsne_true_labels.pdf\n")
  if ("predicted_labels" %in% names(tsne_result$plots)) {
    cat("- tsne_predicted_labels.pdf\n")
    cat("- tsne_prediction_accuracy.pdf\n")
  }
  cat("- tsne_coordinates.csv\n")
  
  return(tsne_result)
}

# Run the analysis
if (!interactive()) {
  result <- main()
}
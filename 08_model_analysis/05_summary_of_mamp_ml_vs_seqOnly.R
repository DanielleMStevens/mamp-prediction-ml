#plot matrix for mamp-ml test data

#load packages
library(readxl, warn.conflicts = FALSE, quietly = TRUE)
library(tidyverse, warn.conflicts = FALSE, quietly = TRUE)
library(ggplot2, warn.conflicts = FALSE, quietly = TRUE)

# Load prediction data
load_mamp_ml_prediction_data <- read.csv("./09_testing_and_dropout/test_data_set/model_00_test_predictions.csv")

# Create confusion matrix
conf_matrix <- table(load_mamp_ml_prediction_data$true_label, 
                    load_mamp_ml_prediction_data$predicted_label)

# Plot confusion matrix
confusion_plot <- ggplot(data = as.data.frame(as.table(conf_matrix)),
                        aes(x = Var2, y = Var1, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "black") +
  scale_fill_gradient(low = "white", high = "grey50") +
  labs(x = "Predicted Label",
       y = "True Label",
       fill = "Count") +
  theme_classic() +
  theme(panel.border = element_rect(color = "black", fill = NA, linewidth = 1)) +
  scale_x_discrete(limits = as.character(0:2)) +
  scale_y_discrete(limits = rev(as.character(0:2)))

ggsave(filename = "./10_alphafold_analysis/Validation_data/confusion_matrix_mamp_ml.pdf", 
       plot = confusion_plot, device = "pdf", dpi = 300, width = 2.4, height = 1.8)


# Load prediction data
load_mamp_ml_prediction_data <- read.csv("./09_testing_and_dropout/test_data_set/model_02_test_predictions.csv")

# Create confusion matrix
conf_matrix <- table(load_mamp_ml_prediction_data$true_label, 
                    load_mamp_ml_prediction_data$predicted_label)

# Plot confusion matrix
confusion_plot <- ggplot(data = as.data.frame(as.table(conf_matrix)),
                        aes(x = Var2, y = Var1, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "black") +
  scale_fill_gradient(low = "white", high = "grey50") +
  labs(x = "Predicted Label",
       y = "True Label",
       fill = "Count") +
  theme_classic() +
  theme(panel.border = element_rect(color = "black", fill = NA, linewidth = 1)) +
  scale_x_discrete(limits = as.character(0:2)) +
  scale_y_discrete(limits = rev(as.character(0:2)))

ggsave(filename = "./10_alphafold_analysis/Validation_data/confusion_matrix_SeqOnly.pdf", 
       plot = confusion_plot, device = "pdf", dpi = 300, width = 2.4, height = 1.8)
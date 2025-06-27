#plot matrix for mamp-ml test data

#load packages
library(readxl, warn.conflicts = FALSE, quietly = TRUE)
library(tidyverse, warn.conflicts = FALSE, quietly = TRUE)
library(ggplot2, warn.conflicts = FALSE, quietly = TRUE)

# Load prediction data
load_mamp_ml_prediction_data <- read.csv("./09_testing_and_dropout/validation_data_set/model_00_test_predictions.csv")

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
  theme(panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5),
  axis.text = element_text(color = "black")) +
  scale_x_discrete(limits = as.character(0:2)) +
  scale_y_discrete(limits = rev(as.character(0:2)))

ggsave(filename = "./10_alphafold_analysis/Validation_data/confusion_matrix_mamp_ml.pdf", 
       plot = confusion_plot, device = "pdf", dpi = 300, width = 2.4, height = 1.8)


# Load prediction data
load_mamp_ml_prediction_data <- read.csv("./09_testing_and_dropout/validation_data_set/model_02_test_predictions.csv")

# Create confusion matrix
conf_matrix <- table(load_mamp_ml_prediction_data$true_label, load_mamp_ml_prediction_data$predicted_label)

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
  theme(panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5),
  axis.text = element_text(color = "black")) +
  scale_x_discrete(limits = as.character(0:2)) +
  scale_y_discrete(limits = rev(as.character(0:2)))

ggsave(filename = "./10_alphafold_analysis/Validation_data/confusion_matrix_SeqOnly.pdf", 
plot = confusion_plot, device = "pdf", dpi = 300, width = 2.4, height = 1.8)


# --- plot softmax probabilities for mamp-ml across all validation data ----

prob_class0_plot <- ggplot(data = melt(subset(load_mamp_ml_prediction_data, load_mamp_ml_prediction_data$true_label == 0)[,1:3]), aes(x = value)) +
facet_wrap(~variable, scales = "free_y", labeller = labeller(variable = c(
  "prob_class0" = "Class 0: Immunogenic",
  "prob_class1" = "Class 1: Non-Immunogenic", 
  "prob_class2" = "Class 2: Weakly Immunogenic"
))) +
geom_density(aes(fill = factor(variable)), alpha = 0.5) +
scale_fill_manual(values = c("prob_class0" = "grey", "prob_class1" = "darkred", "prob_class2" = "darkblue")) +
xlim(0, 1) +
xlab("Probability of True Class: Immunogenic") +
theme_classic() +
theme(panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5),
  axis.text = element_text(color = "black"), legend.position = "none", axis.text.x = element_text(color = "black", size = 7), 
  axis.text.y = element_text(color = "black", size = 7), axis.title.x = element_text(color = "black", size = 7),
  axis.title.y = element_text(color = "black", size = 7), strip.text = element_text(size = 6))

prob_class1_plot <- ggplot(data = melt(subset(load_mamp_ml_prediction_data, load_mamp_ml_prediction_data$true_label == 1)[,1:3]), aes(x = value)) +
facet_wrap(~variable, scales = "free_y", labeller = labeller(variable = c(
  "prob_class0" = "Class 0: Immunogenic",
  "prob_class1" = "Class 1: Non-Immunogenic", 
  "prob_class2" = "Class 2: Weakly Immunogenic"
))) +
geom_density(aes(fill = factor(variable)), alpha = 0.5) +
scale_fill_manual(values = c("prob_class0" = "grey", "prob_class1" = "darkred", "prob_class2" = "darkblue")) +
xlim(0, 1) +
xlab("Probability of True Class: Non-Immunogenic") +
theme_classic() +
theme(panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5),
  axis.text = element_text(color = "black"), legend.position = "none", axis.text.x = element_text(color = "black", size = 7), 
  axis.text.y = element_text(color = "black", size = 7), axis.title.x = element_text(color = "black", size = 7),
  axis.title.y = element_text(color = "black", size = 7), strip.text = element_text(size = 6))

prob_class2_plot <- ggplot(data = melt(subset(load_mamp_ml_prediction_data, load_mamp_ml_prediction_data$true_label == 2)[,1:3]), aes(x = value)) +
facet_wrap(~variable, scales = "free_y", labeller = labeller(variable = c(
  "prob_class0" = "Class 0: Immunogenic",
  "prob_class1" = "Class 1: Non-Immunogenic", 
  "prob_class2" = "Class 2: Weakly Immunogenic"
))) +
geom_density(aes(fill = factor(variable)), alpha = 0.5) +
scale_fill_manual(values = c("prob_class0" = "grey", "prob_class1" = "darkred", "prob_class2" = "darkblue")) +
xlim(0, 1) +
xlab("Probability of True Class: Weakly Immunogenic") +
theme_classic() +
theme(panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5),
  axis.text = element_text(color = "black"), legend.position = "none", axis.text.x = element_text(color = "black", size = 7), 
  axis.text.y = element_text(color = "black", size = 7), axis.title.x = element_text(color = "black", size = 7),
  axis.title.y = element_text(color = "black", size = 7), strip.text = element_text(size = 6))

softmax_prob_plot <- prob_class0_plot / prob_class1_plot / prob_class2_plot

ggsave(filename = "./10_alphafold_analysis/Validation_data/softmax_prob_plot.pdf", 
plot = softmax_prob_plot, device = "pdf", dpi = 300, width = 4.9, height = 4.5)
# Load required libraries
library(readxl)
library(dplyr)
library(ggplot2)
library(pROC)
library(PRROC)
library(reshape2)


######################################################################
# Prediction data from Ngou et al. 2025 to plot zero-shot predictions
######################################################################

# ------------ Ngou et al. 2025 SCORE Ortholog ROS screen data ------------

# Create a data frame with row numbers for x-axis and split into immunogenic categories
data <- readxl::read_excel("./09_testing_and_dropout/Ngou_2025_SCORE_data/Zero_shot_mamp_ml/Ngou_zero_shot_case.xlsx", sheet = "Ngou_Orthologs")

# Now create prediction accuracy column
data$prediction_accuracy <- ifelse(data$`Known Label` == data$`Predicted Label`, "Correct", "Incorrect")

# Calculate counts of correct/incorrect predictions per receptor
prediction_summary <- data %>%
  group_by(locus_id, prediction_accuracy) %>%
  summarise(count = n(), .groups = 'drop')

# Create stacked bar plot
ggplot(prediction_summary, aes(x = locus_id, y = count, fill = prediction_accuracy)) +
  geom_bar(stat = "identity") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Receptor", 
       y = "Count",
       fill = "Prediction Accuracy") +
  scale_fill_manual(values = c("Correct" = "#88CCEE", "Incorrect" = "#CC6677"))

# Create boxplot comparing correct vs incorrect predictions across receptors
ggplot(prediction_summary, aes(y = count, fill = prediction_accuracy)) +
  geom_boxplot(position = position_dodge(width = 0.75)) +
  theme_bw() +
  labs(x = "Receptor",
       y = "Count",
       fill = "Prediction Accuracy") +
  scale_fill_manual(values = c("Correct" = "#2ecc71", "Incorrect" = "#e74c3c"))


# Calculate counts of correct/incorrect predictions per receptor and known label
prediction_summary <- data %>%
  group_by(locus_id, prediction_accuracy, `Known Label`) %>%
  summarise(count = n(), .groups = 'drop')

# Create stacked bar plot showing immunogenic outcomes
ggplot(prediction_summary, aes(x = locus_id, y = count, fill = interaction(`Known Label`, prediction_accuracy))) +
  geom_bar(stat = "identity") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Receptor", 
       y = "Count",
       fill = "Prediction Accuracy") +
  scale_fill_manual(values = c(
    "0.Correct" = "#a8d5e5",    # Muted light blue for correct immunogenic
    "1.Correct" = "#5c97c1",    # Muted medium blue for correct non-immunogenic
    "2.Correct" = "#4a6b84",    # Muted navy for correct weakly immunogenic  
    "0.Incorrect" = "#ffd966",  # Brighter yellow for incorrect immunogenic
    "1.Incorrect" = "#ffb347",  # Brighter orange for incorrect non-immunogenic
    "2.Incorrect" = "#ff8c42"   # Brighter dark orange for incorrect weakly immunogenic
  )) +
  guides(fill = guide_legend(title = "Outcome & Accuracy"))

# Create boxplot comparing correct vs incorrect predictions across immunogenic outcomes
ggplot(prediction_summary, aes(x = factor(`Known Label`), y = count, fill = prediction_accuracy)) +
  geom_boxplot(position = position_dodge(width = 0.75)) +
  theme_bw() +
  labs(x = "Immunogenic Outcome (0=Immunogenic, 1=Non-immunogenic, 2=Weakly)",
       y = "Count",
       fill = "Prediction Accuracy") +
  scale_fill_manual(values = c("Correct" = "#2ecc71", "Incorrect" = "#e74c3c"))

# ------------ Ngou et al. 2025 SCORE LRR Swaps ROS screen data ------------

data <- readxl::read_excel("./09_testing_and_dropout/Ngou_2025_SCORE_data/Zero_shot_mamp_ml/Ngou_zero_shot_case.xlsx", sheet = "Ngou_LRR_Swaps")

# Now create prediction accuracy column
data$prediction_accuracy <- ifelse(data$`Known Label` == data$`Predicted Label`, "Correct", "Incorrect")

# Calculate counts of correct/incorrect predictions per receptor
prediction_summary <- data %>%
  group_by(locus_id, prediction_accuracy) %>%
  summarise(count = n(), .groups = 'drop')

# Create stacked bar plot
ggplot(prediction_summary, aes(x = locus_id, y = count, fill = prediction_accuracy)) +
  geom_bar(stat = "identity") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Receptor", 
       y = "Count",
       fill = "Prediction Accuracy") +
  scale_fill_manual(values = c("Correct" = "#88CCEE", "Incorrect" = "#CC6677"))

# Create boxplot comparing correct vs incorrect predictions across receptors
ggplot(prediction_summary, aes(y = count, fill = prediction_accuracy)) +
  geom_boxplot(position = position_dodge(width = 0.75)) +
  theme_bw() +
  labs(x = "Receptor",
       y = "Count",
       fill = "Prediction Accuracy") +
  scale_fill_manual(values = c("Correct" = "#2ecc71", "Incorrect" = "#e74c3c"))


# Calculate counts of correct/incorrect predictions per receptor and known label
prediction_summary <- data %>%
  group_by(locus_id, prediction_accuracy, `Known Label`) %>%
  summarise(count = n(), .groups = 'drop')

# Create stacked bar plot showing immunogenic outcomes
ggplot(prediction_summary, aes(x = locus_id, y = count, fill = interaction(`Known Label`, prediction_accuracy))) +
  geom_bar(stat = "identity") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Receptor", 
       y = "Count",
       fill = "Prediction Accuracy") +
  scale_fill_manual(values = c(
    "0.Correct" = "#a8d5e5",    # Muted light blue for correct immunogenic
    "1.Correct" = "#5c97c1",    # Muted medium blue for correct non-immunogenic
    "2.Correct" = "#4a6b84",    # Muted navy for correct weakly immunogenic  
    "0.Incorrect" = "#ffd966",  # Brighter yellow for incorrect immunogenic
    "1.Incorrect" = "#ffb347",  # Brighter orange for incorrect non-immunogenic
    "2.Incorrect" = "#ff8c42"   # Brighter dark orange for incorrect weakly immunogenic
  )) +
  guides(fill = guide_legend(title = "Outcome & Accuracy"))

# Create boxplot comparing correct vs incorrect predictions across immunogenic outcomes
ggplot(prediction_summary, aes(x = factor(`Known Label`), y = count, fill = prediction_accuracy)) +
  geom_boxplot(position = position_dodge(width = 0.75)) +
  theme_bw() +
  labs(x = "Immunogenic Outcome (0=Immunogenic, 1=Non-immunogenic, 2=Weakly)",
       y = "Count",
       fill = "Prediction Accuracy") +
  scale_fill_manual(values = c("Correct" = "#2ecc71", "Incorrect" = "#e74c3c"))

# ------------ Ngou et al. 2025 SCORE AA Substitution ROS screen data ------------

data <- readxl::read_excel("./09_testing_and_dropout/Ngou_2025_SCORE_data/Zero_shot_mamp_ml/Ngou_zero_shot_case.xlsx", sheet = "Ngou_AA_Substitutions")

# Now create prediction accuracy column
data$prediction_accuracy <- ifelse(data$`Known Label` == data$`Predicted Label`, "Correct", "Incorrect")

# Calculate counts of correct/incorrect predictions per receptor
prediction_summary <- data %>%
  group_by(locus_id, prediction_accuracy) %>%
  summarise(count = n(), .groups = 'drop')

# Create stacked bar plot
ggplot(prediction_summary, aes(x = locus_id, y = count, fill = prediction_accuracy)) +
  geom_bar(stat = "identity") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Receptor", 
       y = "Count",
       fill = "Prediction Accuracy") +
  scale_fill_manual(values = c("Correct" = "#88CCEE", "Incorrect" = "#CC6677"))

# Create boxplot comparing correct vs incorrect predictions across receptors
ggplot(prediction_summary, aes(y = count, fill = prediction_accuracy)) +
  geom_boxplot(position = position_dodge(width = 0.75)) +
  theme_bw() +
  labs(x = "Receptor",
       y = "Count",
       fill = "Prediction Accuracy") +
  scale_fill_manual(values = c("Correct" = "#2ecc71", "Incorrect" = "#e74c3c"))


# Calculate counts of correct/incorrect predictions per receptor and known label
prediction_summary <- data %>%
  group_by(locus_id, prediction_accuracy, `Known Label`) %>%
  summarise(count = n(), .groups = 'drop')

# Create stacked bar plot showing immunogenic outcomes
ggplot(prediction_summary, aes(x = locus_id, y = count, fill = interaction(`Known Label`, prediction_accuracy))) +
  geom_bar(stat = "identity") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Receptor", 
       y = "Count",
       fill = "Prediction Accuracy") +
  scale_fill_manual(values = c(
    "0.Correct" = "#a8d5e5",    # Muted light blue for correct immunogenic
    "1.Correct" = "#5c97c1",    # Muted medium blue for correct non-immunogenic
    "2.Correct" = "#4a6b84",    # Muted navy for correct weakly immunogenic  
    "0.Incorrect" = "#ffd966",  # Brighter yellow for incorrect immunogenic
    "1.Incorrect" = "#ffb347",  # Brighter orange for incorrect non-immunogenic
    "2.Incorrect" = "#ff8c42"   # Brighter dark orange for incorrect weakly immunogenic
  )) +
  guides(fill = guide_legend(title = "Outcome & Accuracy"))

# Create boxplot comparing correct vs incorrect predictions across immunogenic outcomes
ggplot(prediction_summary, aes(x = factor(`Known Label`), y = count, fill = prediction_accuracy)) +
  geom_boxplot(position = position_dodge(width = 0.75)) +
  theme_bw() +
  labs(x = "Immunogenic Outcome (0=Immunogenic, 1=Non-immunogenic, 2=Weakly)",
       y = "Count",
       fill = "Prediction Accuracy") +
  scale_fill_manual(values = c("Correct" = "#2ecc71", "Incorrect" = "#e74c3c"))

######################################################################
# Prediction data from Ngou et al. 2025 to plot few-shot predictions
######################################################################
# Load required libraries
library(readxl)
library(dplyr)
library(ggplot2)
library(pROC)
library(PRROC)
library(reshape2)
library(tidyverse)

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

prediction_summary$group_type <- 'Orthologs'
prediction_summary_all_groups <- prediction_summary

# Calculate counts of correct/incorrect predictions per receptor and known label
prediction_summary <- data %>%
  group_by(locus_id, prediction_accuracy, `Known Label`) %>%
  summarise(count = n(), .groups = 'drop')

# ------------ Zero-shot plot of ortholog data as a stacked bar plot ------------

# Create stacked bar plot showing immunogenic outcomes
ortholog_stacked_bar_plot_zero_shot <- ggplot(prediction_summary, aes(x = locus_id, y = count, fill = interaction(`Known Label`, prediction_accuracy))) +
  geom_bar(stat = "identity") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, color = "black", size = 7),
        axis.text.y = element_text(color = "black", size = 7),
        axis.title.x = element_text(color = "black", size = 8),
        axis.title.y = element_text(color = "black", size = 8),
        legend.text = element_text(color = "black", size = 7),
        legend.title = element_text(color = "black", size = 6)
      ) +
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

ggsave("ortholog_stacked_bar_plot_zero_shot.pdf", ortholog_stacked_bar_plot_zero_shot, width = 4.5, height = 2.2, dpi = 300, path = "./09_testing_and_dropout/Ngou_2025_SCORE_data/Zero_shot_mamp_ml/", device = "pdf")


# ------------ Zero-shot plot of ortholog data as a boxplot ------------

# Create boxplot comparing correct vs incorrect predictions across immunogenic outcomes
totals <- prediction_summary %>%
  group_by(`Known Label`, prediction_accuracy) %>%
  summarise(total = sum(count))

ortholog_box_plot_zero_shot <- ggplot(prediction_summary, aes(x = factor(`Known Label`), y = count, fill = prediction_accuracy)) +
  geom_boxplot(position = position_dodge(width = 0.75), outlier.shape = NA) +
  geom_text(data = totals, aes(x = as.numeric(`Known Label`) + c(0.8, 1.2), y = 100,
            label = total, group = prediction_accuracy), 
            position = position_dodge(width = 0), size = 2.5) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, color = "black", size = 7),
        axis.text.y = element_text(color = "black", size = 7),
        axis.title.x = element_text(color = "black", size = 8),
        axis.title.y = element_text(color = "black", size = 8),
        legend.position = "none"
      ) +
  ylim(0, 110) +
  labs(x = "",
       y = "Count", 
       fill = "Prediction Accuracy") +
  scale_fill_manual(values = c("Correct" = "#88CCEE", "Incorrect" = "#CC6677")) +
  scale_x_discrete(labels = c("0" = "Immunogenic", "1" = "Non-Immunogenic", "2" = "Weakly Immunogenic"))

ggsave("ortholog_box_plot_zero_shot.pdf", ortholog_box_plot_zero_shot, width = 2, height = 2.2, dpi = 300, path = "./09_testing_and_dropout/Ngou_2025_SCORE_data/Zero_shot_mamp_ml/", device = "pdf")


#################################################################################
# ------------ Ngou et al. 2025 SCORE LRR Swaps ROS screen data ------------
#################################################################################


data <- readxl::read_excel("./09_testing_and_dropout/Ngou_2025_SCORE_data/Zero_shot_mamp_ml/Ngou_zero_shot_case.xlsx", sheet = "Ngou_LRR_Swaps")

# Now create prediction accuracy column
data$prediction_accuracy <- ifelse(data$`Known Label` == data$`Predicted Label`, "Correct", "Incorrect")

# Calculate counts of correct/incorrect predictions per receptor
prediction_summary <- data %>%
  group_by(locus_id, prediction_accuracy) %>%
  summarise(count = n(), .groups = 'drop')

prediction_summary$group_type <- 'LRR_Swaps'
prediction_summary_all_groups <- rbind(prediction_summary_all_groups, prediction_summary)

# Calculate counts of correct/incorrect predictions per receptor and known label
prediction_summary <- data %>%
  group_by(locus_id, prediction_accuracy, `Known Label`) %>%
  summarise(count = n(), .groups = 'drop')

# ------------ Zero-shot plot of lrr_swaps data as a stacked bar plot ------------

# Create stacked bar plot showing immunogenic outcomes
lrr_swaps_stacked_bar_plot_zero_shot <- ggplot(prediction_summary, aes(x = locus_id, y = count, fill = interaction(`Known Label`, prediction_accuracy))) +
  geom_bar(stat = "identity") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, color = "black", size = 7),
        axis.text.y = element_text(color = "black", size = 7),
        axis.title.x = element_text(color = "black", size = 8),
        axis.title.y = element_text(color = "black", size = 8),
        legend.text = element_text(color = "black", size = 7),
        legend.title = element_text(color = "black", size = 6)
      ) +
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

ggsave("lrr_swaps_stacked_bar_plot_zero_shot.pdf", lrr_swaps_stacked_bar_plot_zero_shot, width = 4.5, height = 2.2, dpi = 300, path = "./09_testing_and_dropout/Ngou_2025_SCORE_data/Zero_shot_mamp_ml/", device = "pdf")

# ------------ Zero-shot plot of lrr_swaps data as a boxplot ------------

# Create boxplot comparing correct vs incorrect predictions across immunogenic outcomes
totals <- prediction_summary %>%
  group_by(`Known Label`, prediction_accuracy) %>%
  summarise(total = sum(count))

lrr_swaps_box_plot_zero_shot <- ggplot(prediction_summary, aes(x = factor(`Known Label`), y = count, fill = prediction_accuracy)) +
  geom_boxplot(position = position_dodge(width = 0.75), outlier.shape = NA) +
  geom_text(data = totals, aes(x = as.numeric(`Known Label`) + c(0.8, 1.2), y = 100,
            label = total, group = prediction_accuracy), 
            position = position_dodge(width = 0), size = 2.5) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, color = "black", size = 7),
        axis.text.y = element_text(color = "black", size = 7),
        axis.title.x = element_text(color = "black", size = 8),
        axis.title.y = element_text(color = "black", size = 8),
        legend.position = "none"
      ) +
  ylim(0, 110) +
  labs(x = "",
       y = "Count", 
       fill = "Prediction Accuracy") +
  scale_fill_manual(values = c("Correct" = "#88CCEE", "Incorrect" = "#CC6677")) +
  scale_x_discrete(labels = c("0" = "Immunogenic", "1" = "Non-Immunogenic", "2" = "Weakly Immunogenic"))

ggsave("lrr_swaps_box_plot_zero_shot.pdf", lrr_swaps_box_plot_zero_shot, width = 2, height = 2.2, dpi = 300, path = "./09_testing_and_dropout/Ngou_2025_SCORE_data/Zero_shot_mamp_ml/", device = "pdf")


############################################################################################
# ------------ Ngou et al. 2025 SCORE AA Substitution ROS screen data ------------
############################################################################################

data <- readxl::read_excel("./09_testing_and_dropout/Ngou_2025_SCORE_data/Zero_shot_mamp_ml/Ngou_zero_shot_case.xlsx", sheet = "Ngou_AA_Substitutions")

# Now create prediction accuracy column
data$prediction_accuracy <- ifelse(data$`Known Label` == data$`Predicted Label`, "Correct", "Incorrect")

# Calculate counts of correct/incorrect predictions per receptor
prediction_summary <- data %>%
  group_by(locus_id, prediction_accuracy) %>%
  summarise(count = n(), .groups = 'drop')

prediction_summary$group_type <- 'AA_Sub'
prediction_summary_all_groups <- rbind(prediction_summary_all_groups, prediction_summary)

# Calculate counts of correct/incorrect predictions per receptor and known label
prediction_summary <- data %>%
  group_by(locus_id, prediction_accuracy, `Known Label`) %>%
  summarise(count = n(), .groups = 'drop')

# ------------ Zero-shot plot of aa_subs data as a stacked bar plot ------------
# weird font error I thought I already fixed
prediction_summary[prediction_summary$`Known Label` %in% c("Non-immunogenic"),][3] <- "Non-Immunogenic"
prediction_summary[prediction_summary$`Known Label` %in% c("Weakly immunogenic"),][3] <- "Weakly Immunogenic"


# Create stacked bar plot showing immunogenic outcomes
aa_subs_stacked_bar_plot_zero_shot <- ggplot(prediction_summary, aes(x = locus_id, y = count, fill = interaction(`Known Label`, prediction_accuracy))) +
  geom_bar(stat = "identity") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, color = "black", size = 7),
        axis.text.y = element_text(color = "black", size = 7),
        axis.title.x = element_text(color = "black", size = 8),
        axis.title.y = element_text(color = "black", size = 8),
        legend.text = element_text(color = "black", size = 7),
        legend.title = element_text(color = "black", size = 6)
      ) +
  labs(x = "Receptor", 
       y = "Count",
       fill = "Prediction Accuracy") +
  scale_fill_manual(values = c(
    "Immunogenic.Correct" = "#a8d5e5",    # Muted light blue for correct immunogenic
    "Non-Immunogenic.Correct" = "#5c97c1",    # Muted medium blue for correct non-immunogenic
    "Weakly Immunogenic.Correct" = "#4a6b84",    # Muted navy for correct weakly immunogenic  
    "Immunogenic.Incorrect" = "#ffd966",  # Brighter yellow for incorrect immunogenic
    "Non-Immunogenic.Incorrect" = "#ffb347",  # Brighter orange for incorrect non-immunogenic
    "Weakly Immunogenic.Incorrect" = "#ff8c42"   # Brighter dark orange for incorrect weakly immunogenic
  )) +
  guides(fill = guide_legend(title = "Outcome & Accuracy"))

ggsave("aa_subs_stacked_bar_plot_zero_shot.pdf", aa_subs_stacked_bar_plot_zero_shot, width = 7, height = 2.2, dpi = 300, path = "./09_testing_and_dropout/Ngou_2025_SCORE_data/Zero_shot_mamp_ml/", device = "pdf")

# ------------ Zero-shot plot of lrr_swaps data as a boxplot ------------

# Create boxplot comparing correct vs incorrect predictions across immunogenic outcomes
totals <- prediction_summary %>%
  group_by(`Known Label`, prediction_accuracy) %>%
  summarise(total = sum(count))

aa_subs_box_plot_zero_shot <- ggplot(prediction_summary, aes(x = factor(`Known Label`), y = count, fill = prediction_accuracy)) +
  geom_boxplot(position = position_dodge(width = 0.75), outlier.shape = NA) +
  geom_text(data = totals, aes(x = as.numeric(`Known Label`) + c(0.8, 1.2), y = 100,
            label = total, group = prediction_accuracy), 
            position = position_dodge(width = 0), size = 2.5) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, color = "black", size = 7),
        axis.text.y = element_text(color = "black", size = 7),
        axis.title.x = element_text(color = "black", size = 8),
        axis.title.y = element_text(color = "black", size = 8),
        legend.position = "none"
      ) +
  ylim(0, 110) +
  labs(x = "",
       y = "Count", 
       fill = "Prediction Accuracy") +
  scale_fill_manual(values = c("Correct" = "#88CCEE", "Incorrect" = "#CC6677")) 


  scale_x_discrete(labels = c("0" = "Immunogenic", "1" = "Non-Immunogenic", "2" = "Weakly Immunogenic"))

ggsave("aa_subs_box_plot_zero_shot.pdf", aa_subs_box_plot_zero_shot, width = 2, height = 2.2, dpi = 300, path = "./09_testing_and_dropout/Ngou_2025_SCORE_data/Zero_shot_mamp_ml/", device = "pdf")




############################################################################################
# Zero-shot summary as a boxplot seperated by each group (Orthologs, LRR_Swaps, AA_Subs)
############################################################################################

# Calculate summary statistics for labels
summary_stats <- prediction_summary_all_groups %>%
  group_by(group_type, prediction_accuracy) %>%
  summarise(total = sum(count), .groups = 'drop')

summary_zero_shot_box_plot <- ggplot(prediction_summary_all_groups, aes(x = factor(group_type, levels = c("Orthologs", "LRR_Swaps", "AA_Sub")), 
                                        y = count, fill = prediction_accuracy)) +
  geom_boxplot(position = position_dodge(width = 0.75), outlier.shape = NA) +
  geom_text(data = summary_stats,
            aes(y = max(prediction_summary_all_groups$count) + 15,
                label = total,
                group = prediction_accuracy),
            position = position_dodge(width = 0.75),
            size = 2.5,
            color = "black") +
  theme_bw() +
  ylim(0, 120) +
  labs(x = "Receptor Group Type", y = "Avg. Count per 103 csp22 ligands", 
       fill = "Prediction Accuracy") +
  scale_fill_manual(values = c("Correct" = "#88CCEE", "Incorrect" = "#CC6677")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, color = "black", size = 7),
        axis.text.y = element_text(color = "black", size = 7),
        axis.title.x = element_text(color = "black", size = 8),
        axis.title.y = element_text(color = "black", size = 8),
        legend.position = "none"
  )

ggsave("summary_zero_shot_box_plot.pdf", summary_zero_shot_box_plot, width = 2.5, height = 2.5, dpi = 300, path = "./09_testing_and_dropout/Ngou_2025_SCORE_data/Zero_shot_mamp_ml/", device = "pdf")



######################################################################
# Prediction data from Ngou et al. 2025 to plot few-shot predictions
######################################################################

data <- read_csv("./09_testing_and_dropout/Ngou_2025_SCORE_data/model_25_ortholog_test_predictions.csv")
data$prediction_accuracy <- ifelse(data$true_label == data$predicted_label, "Correct", "Incorrect")
prediction_summary <- data %>% group_by(true_label, prediction_accuracy) %>% summarise(count = n(), .groups = 'drop')
prediction_summary$group_type = "Orthologs"

data2 <- read_csv("./09_testing_and_dropout/Ngou_2025_SCORE_data/model_25_lrr_swap_test_predictions.csv")
data2$prediction_accuracy <- ifelse(data2$true_label == data2$predicted_label, "Correct", "Incorrect")
prediction_summary2 <- data2 %>% group_by(true_label, prediction_accuracy) %>% summarise(count = n(), .groups = 'drop')
prediction_summary2$group_type = "LRR_Swaps"

data3 <- read_csv("./09_testing_and_dropout/Ngou_2025_SCORE_data/model_25_aa_sub_test_predictions.csv")
data3$prediction_accuracy <- ifelse(data3$true_label == data3$predicted_label, "Correct", "Incorrect")
prediction_summary3 <- data3 %>% group_by(true_label, prediction_accuracy) %>% summarise(count = n(), .groups = 'drop')
prediction_summary3$group_type = "AA_Sub"

all_sum <- rbind(prediction_summary, prediction_summary2, prediction_summary3)
total_all_models <- all_sum %>% group_by(group_type, prediction_accuracy) %>% summarise(total = sum(count))
total_all_models$model <- "kshot_128"

total2 <- prediction_summary_all_groups %>% group_by(group_type, prediction_accuracy) %>% summarise(total = sum(count))
total2$model <- "zero-shot"
total_all_models <- rbind(total_all_models, total2)
rm(total2)

data <- read_csv("./09_testing_and_dropout/Ngou_2025_SCORE_data/model_23_ortholog_test_predictions.csv")
data$prediction_accuracy <- ifelse(data$true_label == data$predicted_label, "Correct", "Incorrect")
prediction_summary <- data %>% group_by(true_label, prediction_accuracy) %>% summarise(count = n(), .groups = 'drop')
prediction_summary$group_type = "Orthologs"

data2 <- read_csv("./09_testing_and_dropout/Ngou_2025_SCORE_data/model_23_lrr_swap_test_predictions.csv")
data2$prediction_accuracy <- ifelse(data2$true_label == data2$predicted_label, "Correct", "Incorrect")
prediction_summary2 <- data2 %>% group_by(true_label, prediction_accuracy) %>% summarise(count = n(), .groups = 'drop')
prediction_summary2$group_type = "LRR_Swaps"

data3 <- read_csv("./09_testing_and_dropout/Ngou_2025_SCORE_data/model_23_aa_sub_test_predictions.csv")
data3$prediction_accuracy <- ifelse(data3$true_label == data3$predicted_label, "Correct", "Incorrect")
prediction_summary3 <- data3 %>% group_by(true_label, prediction_accuracy) %>% summarise(count = n(), .groups = 'drop')
prediction_summary3$group_type = "AA_Sub"
all_sum <- rbind(prediction_summary, prediction_summary2, prediction_summary3)

total2 <- all_sum %>% group_by(group_type, prediction_accuracy) %>% summarise(total = sum(count))
total2$model <- "kshot_32"
total_all_models <- rbind(total_all_models, total2)
rm(total2)


# Calculate accuracy percentages for each model and group
accuracy_by_group <- total_all_models %>%
  group_by(model, group_type) %>%
  summarize(
    accuracy = sum(total[prediction_accuracy == "Correct"]) / sum(total) * 100,
    .groups = 'drop'
  ) %>%
  pivot_wider(
    names_from = model,
    values_from = accuracy
  ) %>%
  mutate(
    percent_improvement_128 = kshot_128 - `zero-shot`,
    percent_improvement_32 = kshot_32 - `zero-shot`
  ) %>%
  select(group_type, `zero-shot`, kshot_128, kshot_32, percent_improvement_128, percent_improvement_32)

# Create a long format dataframe for plotting improvements
improvement_data <- accuracy_by_group %>%
  select(group_type, percent_improvement_128, percent_improvement_32) %>%
  pivot_longer(
    cols = c(percent_improvement_128, percent_improvement_32),
    names_to = "model",
    values_to = "improvement"
  ) %>%
  mutate(
    model = case_when(
      model == "percent_improvement_128" ~ "128-shot",
      model == "percent_improvement_32" ~ "32-shot"
    )
  )
  

improvement_data$model <- factor(improvement_data$model, levels = c("32-shot", "128-shot"))
improvement_data$group_type <- factor(improvement_data$group_type, levels = c("Orthologs", "LRR_Swaps", "AA_Sub"))

# Create the bar plot
improvement_data_plot <- ggplot(improvement_data, aes(x = group_type, y = improvement, fill = improvement > 0)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = paste0(round(improvement, 1), "%"), vjust = ifelse(improvement > 0, -0.5, 1.5)), size = 2.5, color = "black") +
  scale_fill_manual(values = c("darkred", "black")) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 7, color = "black"),
    axis.text.y = element_text(size = 7, color = "black"),
    axis.title.x = element_text(size = 8, color = "black"),
    axis.title.y = element_text(size = 8, color = "black"),
    strip.text = element_text(size = 8, color = "black"),
    legend.position = "none") +
  ylim(-30, 70) +
  labs(x = "Receptor Group Type", y = "Percent Improvement Over Zero-Shot") +
  facet_wrap(~model, nrow=1)   

ggsave("Kshot_improvement_data_plot.pdf", improvement_data_plot, width = 2.5, height = 2.5, dpi = 300, path = "./09_testing_and_dropout/Ngou_2025_SCORE_data/Zero_shot_mamp_ml/", device = "pdf")





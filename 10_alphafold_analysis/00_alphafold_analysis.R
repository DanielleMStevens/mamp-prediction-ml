#-----------------------------------------------------------------------------------------------
# Krasileva Lab - Plant & Microbial Biology Department UC Berkeley
# Author: Danielle M. Stevens
# Last Updated: 07/06/2020
# Script Purpose: 
# Inputs: 
# Outputs: 
#-----------------------------------------------------------------------------------------------

######################################################################
#  libraries to load
######################################################################

# compare sequences of eptiopes and receptors to document sequence variation
# Install Bioconductor
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
#BiocManager::install("Biostrings")

#load packages
library(readxl, warn.conflicts = FALSE, quietly = TRUE)
library(tidyverse, warn.conflicts = FALSE, quietly = TRUE)
library(ggplot2, warn.conflicts = FALSE, quietly = TRUE)
library(ggridges, warn.conflicts = FALSE, quietly = TRUE)
library(Biostrings, warn.conflicts = FALSE, quietly = TRUE)
library(pwalign, warn.conflicts = FALSE, quietly = TRUE)

# color code for genera of interest
epitope_colors <- c("#b35c46","#e2b048","#ebd56d","#b9d090","#37a170","#86c0ce","#7d9fc6", "#32527B", "#542a64", "#232232","#D5869D")
names(epitope_colors) <- c("crip21","csp22","elf18","flg22","flgII-28","In11","nlp", "pep-25", "pg", "scoop","screw")

receptor_colors <- c("#b35c46","#e2b048","#ebd56d","#b9d090","#37a170","#86c0ce","#7d9fc6", "#32527B", "#542a64", "#232232","#D5869D")
names(receptor_colors) <- c("CuRe1","CORE","EFR","FLS2","FLS3","INR","RLP23", "PERU", "RLP42", "MIK2","NUT")


######################################################################
#  plot distribution of correct and misclassified predictions
######################################################################

# load data from excel file
load_AF3_data <- readxl::read_xlsx(path = "./10_alphafold_analysis/AF3_validation_data.xlsx")
colnames(load_AF3_data) <- c("TC#","Plant species","Receptor","Locus ID/Genbank","Epitope","Sequence","Receptor Name",
"Receptor Sequence","Known Outcome","empty","pTM","ipTM","Prediction")

# Create summary with negative values for correct predictions
load_AF3_data_summary <- load_AF3_data %>% 
  group_by(Receptor, Prediction) %>% 
  summarise(n = n()) %>%
  mutate(n = ifelse(Prediction == "Correct", -n, n))

load_AF3_data_summary$n[load_AF3_data_summary$Receptor == "INR" & load_AF3_data_summary$Prediction == "Correct"] <- 
  load_AF3_data_summary$n[load_AF3_data_summary$Receptor == "INR" & load_AF3_data_summary$Prediction == "Correct"] + 
  load_AF3_data_summary$n[load_AF3_data_summary$Receptor == "INR-like" & load_AF3_data_summary$Prediction == "Correct"]
load_AF3_data_summary <- load_AF3_data_summary[load_AF3_data_summary$Receptor != "INR-like",]

#organize receptors for plotting to match model results
receptor_order <- c("FLS2", "RLP23", "CORE", "EFR", "MIK2", "RLP42", "PERU",  "INR", "CuRe1", "FLS3", "NUT")
load_AF3_data_summary <- load_AF3_data_summary %>% mutate(Receptor = factor(Receptor, levels = rev(receptor_order))) # rev() for top-down display

# plot distribution of correct and misclassified predictions
AF3_distribution_plot <- ggplot(load_AF3_data_summary, aes(x = n, y = Receptor, fill = Receptor)) +
  geom_col() +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") + # Add line at zero
  scale_x_continuous(name = "Number of Combinations",
          limits = c(-120, 80), breaks = seq(-120, 80, 40), labels = c(120, 80, 40, 0, 40, 80)) +
  labs(y = "Receptor") +
  theme_classic() +
  scale_fill_manual(values = receptor_colors) +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5, color = "black", size = 7),
        axis.text.y = element_text(color = "black", size = 7),
        axis.title.x = element_text(color = "black", size = 8),
        axis.title.y = element_text(color = "black", size = 8),
        plot.title = element_text(hjust = 0.5),
        legend.position = "none",
        plot.margin = unit(c(1, 0.5, 0.5, 0.5), "cm")) + # Add padding (top, right, bottom, left)
   geom_text(aes(label = abs(n), x = n + ifelse(n < 0, -2, 2)),
     hjust = ifelse(load_AF3_data_summary$n < 0, 1, 0), size = 2.5, color = "black") +
   annotate("text", x = -max(abs(load_AF3_data_summary$n)) * 0.8, y = length(receptor_order) + 0.5,
      label = "Correct", hjust = 0, vjust = -1, size = 2.5, color = "black") +
   annotate("text", x = max(abs(load_AF3_data_summary$n)) * 0.8, y = length(receptor_order) + 0.5, 
      label = "Misclassified", hjust = 0.7, vjust = -1, size = 2.5, color = "black") +
   coord_cartesian(clip = "off") # Allows annotations outside plot area

ggsave(filename = "./10_alphafold_analysis/Test_data/AF3_distribution_plot.pdf", plot = AF3_distribution_plot, 
device = "pdf", dpi = 300, width = 2.4, height = 2.6)

######################################################################
#  plot distribution of correct and misclassified predictions based on ipTM and pTM values
######################################################################

iptm_ptm_plot <- ggplot(load_AF3_data, aes(x = pTM, y = ipTM, color = Receptor)) +
  #annotate("rect", xmin = -Inf, xmax = Inf, ymin = 0.8, ymax = Inf, fill = "black", alpha = 0.1, linetype = "dashed") +
  geom_jitter(alpha = 0.75, size = 1.3, stroke = NA) +
  geom_hline(yintercept = 0.8, linetype = "dashed", color = "black") +
  theme_bw() +
  scale_x_continuous(name = "pTM", limits = c(0.8, 0.95), breaks = seq(0.8, 0.95, 0.05), labels = c(0.8, 0.85, 0.9, 0.95)) +
  scale_color_manual(values = receptor_colors) +
  labs(x = "pTM", y = "ipTM") +
  theme(axis.text.x = element_text(color = "black", size = 6),
        axis.text.y = element_text(color = "black", size = 6),
        axis.title.x = element_text(color = "black", size = 7),
        axis.title.y = element_text(color = "black", size = 7),
        plot.title = element_text(hjust = 0.5),
        axis.ticks.length = unit(0.05, "cm"),
        legend.position = "none",
        strip.text = element_text(size = 5.5)) +
  facet_wrap(~factor(`Known Outcome`, levels = c("Immunogenic", "Weakly Immunogenic", "Non-Immunogenic")), ncol = 3)

  ggsave(filename = "./10_alphafold_analysis/Test_data/iptm_ptm_plot.pdf", plot = iptm_ptm_plot, 
device = "pdf", dpi = 300, width = 3.2, height = 1.3)

  iptm_ptm_plot_no_FLS2 <- ggplot(load_AF3_data[load_AF3_data$Receptor != "FLS2",], aes(x = pTM, y = ipTM, color = Receptor)) +
  #annotate("rect", xmin = -Inf, xmax = Inf, ymin = 0.8, ymax = Inf, fill = "black", alpha = 0.1, linetype = "dashed") +
  geom_jitter(alpha = 0.75, size = 1.3, stroke = NA) +
  geom_hline(yintercept = 0.8, linetype = "dashed", color = "black") +
  scale_x_continuous(name = "pTM", limits = c(0.8, 0.95), breaks = seq(0.8, 0.95, 0.05), labels = c(0.8, 0.85, 0.9, 0.95)) +
  theme_bw() +
  scale_color_manual(values = receptor_colors) +
  labs(x = "pTM", y = "ipTM") +
  theme(axis.text.x = element_text(color = "black", size = 6),
        axis.text.y = element_text(color = "black", size = 6),
        axis.title.x = element_text(color = "black", size = 7),
        axis.title.y = element_text(color = "black", size = 7),
        strip.text = element_text(size = 5.5), 
        axis.ticks.length = unit(0.05, "cm"),
  legend.position = "none") +
  facet_wrap(~factor(`Known Outcome`, levels = c("Immunogenic", "Weakly Immunogenic", "Non-Immunogenic")), ncol = 3)

  ggsave(filename = "./10_alphafold_analysis/Test_data/iptm_ptm_plot_no_FLS2.pdf", plot = iptm_ptm_plot_no_FLS2, 
device = "pdf", dpi = 300, width = 3.2, height = 1.3)

# ------------------------------------ same analysis but with independent test data -----------------------------------------


######################################################################
#  plots for Fig. 4E-F
######################################################################

# load data from excel file
load_AF3_data <- readxl::read_xlsx(path = "./10_alphafold_analysis/AF3_new_test_data.xlsx", sheet = "AF3")
load_AF3_data_new_test <- load_AF3_data[,c(9,10,11,12)]
colnames(load_AF3_data_new_test) <- c("Immunogenicity", "pTM", "ipTM", "Prediction")

load_AF3_data_new_test_summary <- load_AF3_data_new_test %>% 
  group_by(Prediction) %>% 
  summarise(n = n())

load_AF3_data_new_test_summary <- reshape2::melt(load_AF3_data_new_test_summary)
load_AF3_data_new_test_summary$variable <- c("AF3")


load_AF3_data_new_test_summary <- rbind(load_AF3_data_new_test_summary,
                                        data.frame("Prediction" = c("Correct", "Misclassified"), 
                                        "variable" = c("mamp-ml", "mamp-ml"),
                                        "value" = c(88, 32)))

load_AF3_data_new_test_summary <- load_AF3_data_new_test_summary %>%
  mutate(value = ifelse(Prediction == "Correct", -value, value))


# ------------- plot number of correct and misclassified predictions for AF3 and mamp-ml ------------
Prediction_approach_AF3_ML_test_data <- ggplot(load_AF3_data_new_test_summary, aes(x = value, y = variable)) +
  geom_col(fill = "grey50", alpha = 0.85) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") + # Add line at zero
  scale_x_continuous(name = "Number of Combinations",
          limits = c(-120, 120), breaks = c(-120, -80, -40, 0, 40, 80, 120), labels = c(120, 80, 40, 0, 40, 80, 120)) +
  labs(y = "Method") +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5, color = "black", size = 7),
        axis.text.y = element_text(color = "black", size = 7),
        axis.title.x = element_text(color = "black", size = 8),
        axis.title.y = element_text(color = "black", size = 8),
        plot.title = element_text(hjust = 0.5),
        legend.position = "none",
        plot.margin = unit(c(1, 0.5, 0.5, 0.5), "cm")) + # Add padding (top, right, bottom, left)
   geom_text(aes(label = abs(value), x = value + ifelse(value < 0, -2, 2)),
     hjust = ifelse(load_AF3_data_new_test_summary$value < 0, 1, 0), size = 2.5, color = "black") +
   annotate("text", x = -max(abs(load_AF3_data_new_test_summary$value)) * 0.8, y = 3.5,
      label = "Correct", hjust = 0, vjust = 0, size = 2.5, color = "black") +
   annotate("text", x = max(abs(load_AF3_data_new_test_summary$value)) * 0.8, y = 3.5,
      label = "Misclassified", hjust = 0.7, vjust = 0, size = 2.5, color = "black") +
   coord_cartesian(clip = "off") # Allows annotations outside plot area


ggsave(filename = "./10_alphafold_analysis/Validation_data/Prediction_approach_AF3_ML_test_data.pdf", plot = Prediction_approach_AF3_ML_test_data, 
device = "pdf", dpi = 300, width = 2.4, height = 1.4)


# ------------- plot AF3 (ipTM and pTM) values to show that correct prediction are only due to overall poor prediction scores ------------

AF3_ipTM_pTM_plot_validation_data <- ggplot(subset(load_AF3_data_new_test, Immunogenicity != "NT"), aes(x = pTM, y = ipTM, color = Prediction)) +
  geom_point(alpha = 0.75, size = 1.3, stroke = NA) +
  facet_wrap(~factor(`Immunogenicity`, levels = c("Immunogenic", "Weakly Immunogenic", "Non-Immunogenic")), ncol = 3) +
  theme_bw() +
  ylim(0,1) +
  geom_hline(yintercept = 0.8, linetype = "dashed", color = "black") +
  scale_x_continuous(name = "pTM", limits = c(0.8, 0.95), breaks = seq(0.8, 0.95, 0.05), labels = c(0.8, 0.85, 0.9, 0.95)) +
  scale_color_manual(values = c("Correct" = "cadetblue", "Misclassified" = "dark red")) +
  labs(x = "pTM", y = "ipTM") +
  theme(axis.text.x = element_text(color = "black", size = 7), axis.text.y = element_text(color = "black", size = 7),
        axis.title.x = element_text(color = "black", size = 8), axis.title.y = element_text(color = "black", size = 8),
        legend.position = "none", strip.text = element_text(size = 5.5),
        legend.text = element_text(size = 7), legend.title = element_text(size = 8), legend.direction = "vertical")

ggsave(filename = "./10_alphafold_analysis/Validation_data/AF3_score_breakdown_data.pdf", plot = AF3_ipTM_pTM_plot_validation_data, 
device = "pdf", dpi = 300, width = 3.5, height = 1.5)


# ------------------------------------ same analysis but with dropout data -----------------------------------------

######################################################################
#  function to calculate 
######################################################################

identity_calc <- function(label_ids, sequence_list, comparison){
  hold_data <- data.frame("query_id" = character(0), "subject_id" = character(0), "comparison" = character(0), "identity" = numeric(0))
  for (i in 1:length(label_ids)){
    for (j in 2:length(label_ids)){
      alignment <- pwalign::pid(pwalign::pairwiseAlignment(sequence_list[i], sequence_list[j], substitutionMatrix = "BLOSUM62", scoreOnly = FALSE))
      hold_data <- rbind(hold_data, data.frame("query_id" = label_ids[i], "subject_id" = label_ids[j], "comparison" = comparison, "identity" = alignment))
    }
  }
  return(hold_data)
}

######################################################################
#  plots for Fig. S11
######################################################################

# load data from excel file
load_AF3_data <- readxl::read_xlsx(path = "./10_alphafold_analysis/AF3_dropout_data.xlsx")
load_AF3_data <- load_AF3_data[,c(1,2,5,6,8,9,10,11,12)]


# --------------------------------- immunogenicity distribution ---------------------------------
drop_out_immunogenicity_dist_plot <- ggplot(load_AF3_data %>% group_by(`Immunogenicity`) %>% summarize(n = n()), aes(x = `Immunogenicity`, y = n)) +
  geom_bar(stat = "identity", fill = "grey50") +
  theme_classic() +
  ylim(0, 35) +
  theme(axis.text.x = element_text(color = "black", size = 7, angle = 45, hjust = 1),
        axis.text.y = element_text(color = "black", size = 7),
        axis.title.x = element_text(color = "black", size = 9),
        axis.title.y = element_text(color = "black", size = 9),
        strip.text = element_text(size = 5.5), 
        axis.ticks.length = unit(0.05, "cm"),
        legend.position = "none") +
  xlab("") +
  ylab("Count") +
  geom_text(aes(label = n), vjust = -0.5, size = 2)

ggsave(filename = "./10_alphafold_analysis/Dropout_data/drop_out_immunogenicity_dist_plot.pdf", plot = drop_out_immunogenicity_dist_plot, 
device = "pdf", dpi = 300, width = 1.5, height = 1.9)

# pepr receptor
PEPR_comparison <- identity_calc(load_AF3_data$`Ectodomain Sequence`, load_AF3_data$`Ectodomain Sequence`, "PEPR")
PEPR_comparison <- subset(PEPR_comparison, query_id != subject_id)

# pep ligand comparison
pep_comparison <- identity_calc(load_AF3_data$`Ligand Sequence`, load_AF3_data$`Ligand Sequence`, "pep")
pep_comparison <- subset(pep_comparison, query_id != subject_id)

# --------------------------------- PEPR comparison ---------------------------------
PEPR_comparison_plot <- ggplot(PEPR_comparison, aes(x = comparison, y = identity)) +
  stat_ydensity(fill = "grey50", alpha = 0.85, scale = "width") +
  geom_boxplot(fill = "white", width = 0.25, outlier.shape = NA) +
  theme_classic() +
  xlab("") +
  ylab("Percent Identity")+
  theme(axis.text.x = element_text(angle = 45, hjust = 1, color = "black"), 
        axis.text.y = element_text(color = "black"),
        legend.position = "none") +
  scale_y_continuous(limits = c(0, 120), breaks = c(20,40,60,80,100)) +
  #geom_text(data = receptor_stats, aes(x = comparison, y = 110, label = number), size = 3)
  coord_flip() 

ggsave(filename = "./10_alphafold_analysis/Dropout_data/PEPR_comparison_plot.pdf", plot = PEPR_comparison_plot, 
device = "pdf", dpi = 300, width = 2, height = 1)

# --------------------------------- pep comparison ---------------------------------
pep_comparison_plot <- ggplot(pep_comparison, aes(x = comparison, y = identity)) +
  stat_ydensity(fill = "grey50", alpha = 0.85, scale = "width") +
  geom_boxplot(fill = "white", width = 0.25, outlier.shape = NA) +
  theme_classic() +
  xlab("") +
  ylab("Percent Identity")+
  theme(axis.text.x = element_text(angle = 45, hjust = 1, color = "black"), 
        axis.text.y = element_text(color = "black"),
        legend.position = "none") +
  scale_y_continuous(limits = c(0, 120), breaks = c(20,40,60,80,100)) +
  #geom_text(data = receptor_stats, aes(x = comparison, y = 110, label = number), size = 3)
  coord_flip() 

  ggsave(filename = "./10_alphafold_analysis/Dropout_data/pep_comparison_plot.pdf", plot = pep_comparison_plot, 
device = "pdf", dpi = 300, width = 2, height = 1)

######################################################################
#  load prediction data from MAMP-ML and plot confusion matrix
######################################################################

# load data from excel file
# Load prediction data
load_mamp_ml_prediction_data <- read.csv("./09_testing_and_dropout/dropout_case/model_00_dropout_predictions.csv")

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

ggsave(filename = "./10_alphafold_analysis/Dropout_data/confusion_matrix.pdf", 
       plot = confusion_plot, device = "pdf", dpi = 300, width = 2.4, height = 1.8)



# load SeqOnly model data
load_SeqOnly_prediction_data <- read.csv("./09_testing_and_dropout/dropout_case/model_02_dropout_predictions.csv")

# Create confusion matrix
conf_matrix_SeqOnly <- table(load_SeqOnly_prediction_data$true_label, 
                    load_SeqOnly_prediction_data$predicted_label)


# Plot confusion matrix
confusion_plot_SeqOnly <- ggplot(data = as.data.frame(as.table(conf_matrix_SeqOnly)),
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

ggsave(filename = "./10_alphafold_analysis/Dropout_data/confusion_plot_SeqOnly.pdf", 
       plot = confusion_plot_SeqOnly, device = "pdf", dpi = 300, width = 2.4, height = 1.8)

######################################################################
#  load prediction data from AF3 and MAMP-ML and make comparison plot
######################################################################

load_AF3_data_dropout_summary <- load_AF3_data %>% 
  group_by(`Prediction based on 0.8 ipTM Cutoff`) %>% 
  summarise(n = n()) %>%
  mutate(n = ifelse(`Prediction based on 0.8 ipTM Cutoff` == "Correct", -n, n))

colnames(load_AF3_data_dropout_summary) <- c("Prediction", "n")
load_AF3_data_dropout_summary$method <- "AF3"
load_AF3_data_dropout_summary <- rbind(load_AF3_data_dropout_summary, data.frame("Prediction" = "Correct",
                          "n" = -(sum(diag(conf_matrix))),
                          "method" = "mamp-ml"))

load_AF3_data_dropout_summary <- rbind(load_AF3_data_dropout_summary, data.frame("Prediction" = "Misclassified",
                          "n" = sum(conf_matrix) - sum(diag(conf_matrix)),
                          "method" = "mamp-ml"))

load_AF3_data_dropout_summary <- rbind(load_AF3_data_dropout_summary, data.frame("Prediction" = "Correct",
                          "n" = -(sum(diag(conf_matrix_SeqOnly))),
                          "method" = "SeqOnly"))

load_AF3_data_dropout_summary <- rbind(load_AF3_data_dropout_summary, data.frame("Prediction" = "Misclassified",
                          "n" = sum(conf_matrix_SeqOnly) - sum(diag(conf_matrix_SeqOnly)),
                          "method" = "SeqOnly"))

# attach mamp-ml prediction data results and plot
Prediction_approach_AF3_ML_dropout <- ggplot(load_AF3_data_dropout_summary, aes(x = n, y = method)) +
  geom_col(fill = "grey50", alpha = 0.85) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") + # Add line at zero
  scale_x_continuous(name = "Number of Combinations",
          limits = c(-80, 80), breaks = c(-80, -40, 0, 40, 80), labels = c(80, 40, 0, 40, 80)) +
  labs(y = "Method") +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5, color = "black", size = 7),
        axis.text.y = element_text(color = "black", size = 7),
        axis.title.x = element_text(color = "black", size = 8),
        axis.title.y = element_text(color = "black", size = 8),
        plot.title = element_text(hjust = 0.5),
        legend.position = "none",
        plot.margin = unit(c(1, 0.5, 0.5, 0.5), "cm")) + # Add padding (top, right, bottom, left)
   geom_text(aes(label = abs(n), x = n + ifelse(n < 0, -2, 2)),
     hjust = ifelse(load_AF3_data_dropout_summary$n < 0, 1, 0), size = 2.5, color = "black") +
   annotate("text", x = -max(abs(load_AF3_data_dropout_summary$n)) * 0.8, y = 3.5,
      label = "Correct", hjust = 0, vjust = 0, size = 2.5, color = "black") +
   annotate("text", x = max(abs(load_AF3_data_dropout_summary$n)) * 0.8, y = 3.5,
      label = "Misclassified", hjust = 0.7, vjust = 0, size = 2.5, color = "black") +
   coord_cartesian(clip = "off") # Allows annotations outside plot area


ggsave(filename = "./10_alphafold_analysis/Dropout_data/Prediction_approach_AF3_ML_dropout.pdf", plot = Prediction_approach_AF3_ML_dropout, 
device = "pdf", dpi = 300, width = 2.2, height = 1.6)
#-----------------------------------------------------------------------------------------------
# Author: Danielle M. Stevens
# Last Updated: 07/06/2020
# Script Purpose: 
# Inputs: 
# Outputs: 
#-----------------------------------------------------------------------------------------------

######################################################################
#  libraries to load
######################################################################

#load packages
library(readr)
library(tidyverse)
library(ggplot2)
library(ggrepel)

# color code for genera of interest
epitope_colors <- c("#b35c46","#e2b048","#ebd56d","#b9d090","#37a170","#86c0ce","#7d9fc6", "#32527B", "#542a64", "#232232","#D5869D")
names(epitope_colors) <- c("crip21","csp22","elf18","flg22","flgII-28","In11","nlp", "pep-25", "pg", "scoop","screw")

receptor_colors <- c("#b35c46","#e2b048","#ebd56d","#b9d090","#37a170","#86c0ce", "#86c0ce","#7d9fc6", "#32527B", "#542a64", "#232232","#D5869D")
names(receptor_colors) <- c("CuRe1","CORE","EFR","FLS2","FLS3","INR", "INR-like","RLP23", "PERU", "RLP42", "MIK2","NUT")

# load alphafold scores
alphafold_scores <- read_table("./04_Preprocessing_results/alphafold_scores.txt", col_names = TRUE)
colnames(alphafold_scores) <- c("Receptor", "Best_Model", "pLDDT", "pTM")
#summary(alphafold_scores)

######################################################################
#  plots of alphafold scores before processing in LRR-Annotation
######################################################################

# Histogram of pLDDT scores
pLDDT_plot <- ggplot(alphafold_scores, aes(x = pLDDT)) +
            geom_histogram(bins = 10, fill = "steelblue", color = "black") +
            labs(title = "Distribution of pLDDT Scores",x = "pLDDT Score", y = "Count") +
            theme_classic() +
            theme(axis.text.x = element_text(size = 7, color = "black"),
                    axis.text.y = element_text(size = 7, color = "black"),
                    title = element_text(size = 6)) +
            scale_x_continuous(breaks = 80:100)

ggsave(filename = "./04_Preprocessing_results/pLDDT_plot.pdf", plot = pLDDT_plot, device = "pdf", dpi = 300, width = 2.5, height = 2)


# Histogram of pTM scores
pTM_plot <- ggplot(alphafold_scores, aes(x = pTM)) +
            geom_histogram(bins = 10, fill = "darkgreen", color = "black") +
    labs(title = "Distribution of pTM Scores", x = "pTM Score", y = "Count") +
    theme_classic() +
    theme(axis.text.x = element_text(size = 7, color = "black"),
        axis.text.y = element_text(size = 7, color = "black"),
        title = element_text(size = 6)) 
    scale_x_continuous(breaks = 0.45:1.00)

ggsave(filename = "./04_Preprocessing_results/pTM_plot.pdf", plot = pTM_plot, device = "pdf", dpi = 300, width = 2.5, height = 2)


# Scatter plot of pLDDT vs pTM
alphafold_scores$Protein_Family <- str_extract(alphafold_scores$Receptor, "(?<=_)[^_]+$")
pLDDT_pTM_scatter<- ggplot(alphafold_scores, aes(x = pLDDT, y = pTM, color = Protein_Family)) +
  geom_point(stroke = NA, alpha = 0.7) +
  labs( x = "pLDDT Score", y = "pTM Score") +
  scale_color_manual(values = receptor_colors) +
  theme_classic() +
  theme(axis.text.x = element_text(size = 8, color = "black"),
        axis.text.y = element_text(size = 8, color = "black"),
        legend.position = "none") +
    xlim(83, 95) +
    ylim(0.45, 1.00) 

ggsave(filename = "./04_Preprocessing_results/pLDDT_pTM_scatter.pdf", plot = pLDDT_pTM_scatter, device = "pdf", dpi = 300, width = 2.5, height = 2)

######################################################################
#  plot comparisons of LRR-Annotation and literature descriptions
######################################################################

load_training_ML_data <- readxl::read_xlsx(path = "./02_in_data/All_LRR_PRR_ligand_data.xlsx")
load_training_ML_data <- data.frame(load_training_ML_data)[1:12]
load_training_ML_data$Protein_key_LRR_Annotation <- str_replace_all(paste(load_training_ML_data$Plant.species, load_training_ML_data$Locus.ID.Genbank, load_training_ML_data$Receptor, sep = "_"), " ", "_")

# load LRR-Annotation data and summarize max winding and max LRR repeat number
load_lrr_annotation_data <- as.data.frame(read_csv(file = "./04_Preprocessing_results/bfactor_winding_lrr_segments.csv"))
colnames(load_lrr_annotation_data) <- c("Protein_key", "Residue_Index","Filtered_B-factor", "Winding_Number", "LRR_Repeat_Number")
#load_lrr_annotation_data$Residue_Index <- as.integer(load_lrr_annotation_data$Residue_Index)
#load_lrr_annotation_data$`Filtered_B-factor` <- as.numeric(load_lrr_annotation_data$`Filtered_B-factor`)

# ----------- plot b-factor examples for counting lrrs -----------

# Example 1: Solanum lycopersicum Solyc03g096190 CORE
Solanum_lycopersicum_Solyc03g096190_CORE_Bfactor <- ggplot(subset(load_lrr_annotation_data, Protein_key == "Solanum_lycopersicum_Solyc03g096190_CORE"), aes(x = Residue_Index, y = `Filtered_B-factor`)) +
  geom_hline(yintercept = 0, color = "black", linetype = "solid") +
  geom_point(alpha = 0.5, color = "#e2b048", size = 0.7) +
  theme_classic() +
  ylim(-1, 1) +
  labs(x = "Residue Index", y = "B-factor") +
  theme(panel.border = element_rect(color = "black", fill = NA),
        axis.text.x = element_text(size = 8, color = "black"),
        axis.text.y = element_text(size = 8, color = "black"),
        axis.title.x = element_text(size = 9, color = "black"),
        axis.title.y = element_text(size = 9, color = "black"))

ggsave(filename = "./04_Preprocessing_results/CORE_Bfactor.pdf", plot = Solanum_lycopersicum_Solyc03g096190_CORE_Bfactor, device = "pdf", dpi = 300, width = 3, height = 1)

# Example 2: Vigna unguiculata Vigun07g219600 INR
Vigna_unguiculata_Vigun07g219600_INR_Bfactor <- ggplot(subset(load_lrr_annotation_data, Protein_key == "Vigna_unguiculata_Vigun07g219600_INR"), aes(x = Residue_Index, y = `Filtered_B-factor`)) +
  geom_hline(yintercept = 0, color = "black", linetype = "solid") +
  geom_point(alpha = 0.5, color = "#86c0ce", size = 0.7) +
  theme_classic() +
  ylim(-1, 1) +
  labs(x = "Residue Index", y = "B-factor") +
  theme(panel.border = element_rect(color = "black", fill = NA),
        axis.text.x = element_text(size = 8, color = "black"),
        axis.text.y = element_text(size = 8, color = "black"),
        axis.title.x = element_text(size = 9, color = "black"),
        axis.title.y = element_text(size = 9, color = "black"))

ggsave(filename = "./04_Preprocessing_results/INR_Bfactor.pdf", plot = Vigna_unguiculata_Vigun07g219600_INR_Bfactor, device = "pdf", dpi = 300, width = 3, height = 1)


# ----------- plot max winding number vs described number of LRRs -----------

load_lrr_annotation_data <- load_lrr_annotation_data %>% group_by(Protein_key) %>% summarize(max_winding = max(Winding_Number),max_lrr = max(LRR_Repeat_Number))
load_lrr_annotation_data$max_winding <- as.integer(load_lrr_annotation_data$max_winding)
load_training_ML_data <- right_join(load_training_ML_data, load_lrr_annotation_data, by = c("Protein_key_LRR_Annotation" = "Protein_key"))
load_training_ML_data <- load_training_ML_data %>% dplyr::distinct(Protein_key_LRR_Annotation, .keep_all = TRUE)

# plot max winding number vs described number of LRRs
max_winding_vs_lrrs <- ggplot(load_training_ML_data, aes(x = Number.of.LRRs, y = max_winding, color = Receptor, shape = Receptor.Type)) + 
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey", alpha = 0.5) +
  geom_jitter(alpha = 0.7, size = 1.5) + 
  xlim(16,32) + ylim(16,32) +
  labs(x = "Described Number of LRRs", y = "Maximum Winding Number") + 
  theme_classic() +
  theme(legend.position = "none",
        axis.text.x = element_text(size = 8, color = "black"),
        axis.text.y = element_text(size = 8, color = "black"),
        axis.title.x = element_text(size = 9, color = "black"),
        axis.title.y = element_text(size = 9, color = "black"),
        aspect.ratio = 1,
        panel.border = element_rect(color = "black", fill = NA),
        panel.grid.minor = element_line(color = "grey90", size = 0.2),
        panel.grid.major = element_line(color = "grey90", size = 0.2)) +
  scale_color_manual(values = receptor_colors) 

ggsave(filename = "./04_Preprocessing_results/max_winding_vs_lrrs.pdf", plot = max_winding_vs_lrrs, device = "pdf", dpi = 300, width = 2.5, height = 2)

# plot LRR repeat number vs described number of LRRs
lrr_repeat_vs_lrrs <- ggplot(load_training_ML_data, aes(x = Number.of.LRRs, y = max_lrr, color = Receptor, shape = Receptor.Type)) + 
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey", alpha = 0.6) +
  geom_jitter(alpha = 0.65, size = 1.5) + 
  #xlim(16,30) + ylim(16,30) +
  labs(x = "Described Number of LRRs", y = "Predicted Number of LRRs") + 
  theme_classic() +
  theme(legend.position = "none",
        axis.text.x = element_text(size = 8, color = "black"),
        axis.text.y = element_text(size = 8, color = "black"),
        axis.title.x = element_text(size = 9, color = "black"),
        axis.title.y = element_text(size = 9, color = "black"),
        aspect.ratio = 1,
        panel.border = element_rect(color = "black", fill = NA),
        panel.grid.minor = element_line(color = "grey90", size = 0.2),
        panel.grid.major = element_line(color = "grey90", size = 0.2)) +
  scale_color_manual(values = receptor_colors) 

ggsave(filename = "./04_Preprocessing_results/lrr_repeat_vs_lrrs.pdf", plot = lrr_repeat_vs_lrrs, device = "pdf", dpi = 300, width = 2.5, height = 2)
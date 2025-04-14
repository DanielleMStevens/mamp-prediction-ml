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

# color code for genera of interest
epitope_colors <- c("#b35c46","#e2b048","#ebd56d","#b9d090","#37a170","#86c0ce","#7d9fc6", "#2a3c64", "#542a64", "#232232")
names(epitope_colors) <- c("Crip21","csp22","elf18","flg22","flgII-28","In11","nlp", "Pep-25", "pep/pg", "SCOOP")

receptor_colors <- c("#b35c46","#e2b048","#ebd56d","#b9d090","#37a170","#86c0ce","#7d9fc6", "#2a3c64", "#542a64", "#232232")
names(receptor_colors) <- c("CuRe1","CORE","EFR","FLS2","FLS3","INR","RLP23", "PERU", "RLP42", "MIK2")

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


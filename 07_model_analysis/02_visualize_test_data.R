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
epitope_colors <- c("#b35c46","#e2b048","#ebd56d","#b9d090","#37a170","#86c0ce","#7d9fc6", "#2a3c64", "#542a64", "#232232")
names(epitope_colors) <- c("crip21","csp22","elf18","flg22","flgII-28","In11","nlp", "pep-25", "pg", "scoop")

receptor_colors <- c("#b35c46","#e2b048","#ebd56d","#b9d090","#37a170","#86c0ce","#7d9fc6", "#2a3c64", "#542a64", "#232232")
names(receptor_colors) <- c("CuRe1","CORE","EFR","FLS2","FLS3","INR","RLP23", "PERU", "RLP42", "MIK2")

######################################################################
#  plot of test data 
######################################################################

test_data <- readr::read_csv(file = "./05_datasets/test_stratify.csv")
test_summary <- test_data %>% group_by(Receptor) %>% summarize(n = n())
test_summary$test_data <- "test_data"

# Combine INR and INR-like counts and remove INR-like
test_summary$n[test_summary$Receptor == "INR"] <- test_summary$n[test_summary$Receptor == "INR"] + 
test_summary$n[test_summary$Receptor == "INR-like"]
test_summary <- test_summary[test_summary$Receptor != "INR-like",]

# plot a stacked bar chart of the number of sequences in each Receptor
test_summary_plot <-ggplot(test_summary, aes(x = test_data, y = n, fill = Receptor)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = n, color = Receptor),
            size = 2, 
            position = position_stack(vjust = 0.2),
            hjust = -0.2) +
  theme_classic() +
  theme(axis.text.x = element_blank(),
        axis.text.y = element_text(size = 8, color = "black"),
        legend.position = "none") +
  scale_fill_manual(values = receptor_colors) +
  scale_color_manual(values = c(
    "CuRe1" = "black",    # Light color - black text
    "CORE" = "black",     # Light color - black text  
    "EFR" = "black",      # Light color - black text
    "FLS2" = "black",     # Light color - black text
    "FLS3" = "black",     # Light color - black text
    "INR" = "black",      # Light color - black text
    "RLP23" = "white",    # Dark color - white text
    "PERU" = "white",     # Dark color - white text
    "RLP42" = "white",    # Dark color - white text
    "MIK2" = "white"      # Dark color - white text
  )) +
  coord_flip() +
  xlab("") +
  ylab("Count")


ggsave(filename = "./04_Preprocessing_results/test_summary_plot.pdf", plot = test_summary_plot, device = "pdf", dpi = 300, width = 3.2, height = 0.8)
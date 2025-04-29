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
library(Biostrings)
library(pwalign)

# color code for genera of interest
epitope_colors <- c("#b35c46","#e2b048","#ebd56d","#b9d090","#37a170","#86c0ce","#7d9fc6", "#32527B", "#542a64", "#232232","#D5869D")
names(epitope_colors) <- c("crip21","csp22","elf18","flg22","flgII-28","In11","nlp", "pep-25", "pg", "scoop","screw")

receptor_colors <- c("#b35c46","#e2b048","#ebd56d","#b9d090","#37a170","#86c0ce", "#86c0ce","#7d9fc6", "#32527B", "#542a64", "#232232","#D5869D")
names(receptor_colors) <- c("CuRe1","CORE","EFR","FLS2","FLS3","INR", "INR-like","RLP23", "PERU", "RLP42", "MIK2","NUT")


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
#  plot of test data 
######################################################################

test_data <- readr::read_csv(file = "./05_datasets/test_stratify.csv")
test_summary <- test_data %>% group_by(Receptor) %>% summarize(n = n())
test_summary$validation_data <- "validation_data"

# Combine INR and INR-like counts and remove INR-like
test_summary$n[test_summary$Receptor == "INR"] <- test_summary$n[test_summary$Receptor == "INR"]  
test_summary$n[test_summary$Receptor == "INR-like"]
test_summary <- test_summary[test_summary$Receptor != "INR-like",]

# plot a stacked bar chart of the number of sequences in each Receptor
test_summary_plot <- ggplot(test_summary, aes(x = validation_data, y = n, fill = Receptor)) +
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
    "NUT" = "black",      # Light color - black text
    "RLP23" = "white",    # Dark color - white text
    "PERU" = "white",     # Dark color - white text
    "RLP42" = "white",    # Dark color - white text
    "MIK2" = "white"      # Dark color - white text
  )) +
  xlab("") +
  ylab("Count")


ggsave(filename = "./04_Preprocessing_results/Validation_data_plots/test_summary_plot.pdf", plot = test_summary_plot, device = "pdf", dpi = 300, width = 1.2, height = 2.5)


######################################################################
#  plot validation data by peptide immunogenicity
######################################################################

# distribution of peptide outcomes
test_data <- readr::read_csv(file = "./05_datasets/test_stratify.csv")
peptide_distrubution <- test_data %>% group_by(Epitope, `Known Outcome`) %>% summarize(n=n())
immunogenicity_distrubution <- ggplot(data = peptide_distrubution, aes(x=`Known Outcome`, y=n, fill=Epitope)) +
  geom_bar(stat="identity") +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8, color = "black"),
        axis.text.y = element_text(color = "black"),
        axis.title.y = element_blank(),
        legend.position = "none")+
  ggtitle("Validation Data") +
  labs(x="", y="Count") +
  coord_flip() +
  scale_fill_manual(values = epitope_colors) +
  geom_text(data = peptide_distrubution %>% 
              group_by(`Known Outcome`) %>% 
              summarise(n = sum(n)),
            aes(label = n, y = n, x = `Known Outcome`), 
            position = position_stack(vjust = 1.05),
            inherit.aes = FALSE,
            size = 3)

ggsave(filename = "./04_Preprocessing_results/Validation_data_plots/immunogenicity_distrubution.pdf", plot = immunogenicity_distrubution, device = "pdf", dpi = 300, width = 2.8, height = 1.8)



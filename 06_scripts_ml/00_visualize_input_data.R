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
library(readxl)
library(tidyverse)
library(Biostrings)
library(pwalign)
library(ggplot2)

# color code for genera of interest
epitope_colors <- c("#b35c46","#e2b048","#ebd56d","#b9d090","#37a170","#86c0ce","#7d9fc6", "#2a3c64", "#542a64", "#232232")
names(epitope_colors) <- c("Crip21","csp22","elf18","flg22","flgII-28","In11","nlp", "Pep-25", "pep/pg", "SCOOP")

receptor_colors <- c("#b35c46","#e2b048","#ebd56d","#b9d090","#37a170","#86c0ce","#7d9fc6", "#2a3c64", "#542a64", "#232232")
names(receptor_colors) <- c("CuRe1","CORE","EFR","FLS2","FLS3","INR","RLP23", "PERU", "RLP42", "MIK2")

load_training_ML_data <- readxl::read_xlsx(path = "./02_in_data/All_LRR_PRR_ligand_data.xlsx")
load_training_ML_data <- data.frame(load_training_ML_data)[1:12]


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
#  libraries to load
######################################################################

# distribution of peptide outcomes
peptide_distrubution <- load_training_ML_data %>% group_by(Ligand, Immunogenicity) %>% summarize(n=n())
immunogenicity_distrubution <- ggplot(peptide_distrubution, aes(x=Immunogenicity, y=n, fill=Ligand)) +
  geom_bar(stat="identity") +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8, color = "black"),
        axis.text.y = element_text(color = "black"),
        legend.position = "none")+
  labs(x="", y="Count") +
  scale_fill_manual(values = epitope_colors)

ggsave(filename = "./04_Preprocessing_results/peptide_distrubution.pdf", plot = immunogenicity_distrubution, device = "pdf", dpi = 300, width = 1.5, height = 2.5)


######################################################################
#  parse and compare homologs of receptors
######################################################################

# subset training data into fasta files based on receptor/ligand
receptors_list <- unique(load_training_ML_data$Receptor)

# inr receptor
receptor_INR <- rbind(load_training_ML_data %>% filter(Receptor == receptors_list[1]), load_training_ML_data %>% filter(Receptor == receptors_list[2]))
receptor_INR <- receptor_INR[!duplicated(receptor_INR$Receptor.Sequence),]
INR_comparison <- identity_calc(receptor_INR$Locus.ID.Genbank, receptor_INR$Receptor.Sequence, "INR")
INR_comparison <- subset(INR_comparison, query_id != subject_id)

# fls2 receptor
receptor_FLS2 <- load_training_ML_data %>% filter(Receptor == receptors_list[3])
receptor_FLS2 <- receptor_FLS2[!duplicated(receptor_FLS2$Receptor.Sequence),]
FLS2_comparison <- identity_calc(receptor_FLS2$Locus.ID.Genbank, receptor_FLS2$Receptor.Sequence, "FLS2")
FLS2_comparison <- subset(FLS2_comparison, query_id != subject_id)

# peru receptor
receptor_PERU <- load_training_ML_data %>% filter(Receptor == receptors_list[4])
receptor_PERU <- receptor_PERU[!duplicated(receptor_PERU$Receptor.Sequence),]
PERU_comparison <- identity_calc(receptor_PERU$Locus.ID.Genbank, receptor_PERU$Receptor.Sequence, "PERU")
PERU_comparison <- subset(PERU_comparison, query_id != subject_id)

# mik2 receptor
receptor_MIK2 <- load_training_ML_data %>% filter(Receptor == receptors_list[5])
receptor_MIK2 <- receptor_MIK2[!duplicated(receptor_MIK2$Receptor.Sequence),]
MIK2_comparison <- identity_calc(receptor_MIK2$Locus.ID.Genbank, receptor_MIK2$Receptor.Sequence, "MIK2")
MIK2_comparison <- subset(MIK2_comparison, query_id != subject_id)

# core receptor
receptor_CORE <- load_training_ML_data %>% filter(Receptor == receptors_list[6])
receptor_CORE <- receptor_CORE[!duplicated(receptor_CORE$Receptor.Sequence),]
CORE_comparison <- identity_calc(receptor_CORE$Locus.ID.Genbank, receptor_CORE$Receptor.Sequence, "CORE")
CORE_comparison <- subset(CORE_comparison, query_id != subject_id)

# efr receptor
receptor_EFR <- load_training_ML_data %>% filter(Receptor == receptors_list[7])
receptor_EFR <- receptor_EFR[!duplicated(receptor_EFR$Receptor.Sequence),]
EFR_comparison <- identity_calc(receptor_EFR$Locus.ID.Genbank, receptor_EFR$Receptor.Sequence, "EFR")
EFR_comparison <- subset(EFR_comparison, query_id != subject_id)

# fls3 receptor
receptor_FLS3 <- load_training_ML_data %>% filter(Receptor == receptors_list[8])
receptor_FLS3 <- receptor_FLS3[!duplicated(receptor_FLS3$Receptor.Sequence),]
FLS3_comparison <- identity_calc(receptor_FLS3$Locus.ID.Genbank, receptor_FLS3$Receptor.Sequence, "FLS3")
FLS3_comparison <- subset(FLS3_comparison, query_id != subject_id)


combine_receptor_comparison <- rbind(FLS2_comparison, PERU_comparison, MIK2_comparison, CORE_comparison, EFR_comparison, FLS3_comparison, INR_comparison)
receptor_stats <- combine_receptor_comparison %>% group_by(comparison) %>%distinct(query_id) %>% summarize(number = n())
receptor_sequence_comparison_plot <- ggplot(combine_receptor_comparison, aes(x = comparison, y = identity, fill = comparison)) +
  stat_ydensity(aes(color = comparison), alpha = 0.85, scale = "width") +
  geom_boxplot(fill = "white", width = 0.2, outlier.shape = NA) +
  theme_classic() +
  xlab("Homolog Comparisons") +
  ylab("Percent Identity")+
  theme(axis.text.x = element_text(angle = 45, hjust = 1, color = "black"), 
        axis.text.y = element_text(color = "black"),
        legend.position = "none") +
  scale_y_continuous(limits = c(20, 120), breaks = c(20,40,60,80,100)) +
  coord_flip() +
  scale_fill_manual(values = receptor_colors) +
  scale_color_manual(values = receptor_colors) +
  geom_text(data = receptor_stats, aes(x = comparison, y = 110, label = number), size = 3)

ggsave(filename = "./04_Preprocessing_results/receptor_sequence_comparison_plot.pdf", plot = receptor_sequence_comparison_plot, device = "pdf", dpi = 300, width = 2.5, height = 2.5)


######################################################################
#  parse and compare epitope variants
######################################################################

# subset training data into fasta files based on receptor/ligand
epitope_list <- unique(load_training_ML_data$Ligand)

# flg22 epitope
epitope_flg22 <- load_training_ML_data %>% filter(Ligand == epitope_list[2])
epitope_flg22 <- epitope_flg22[!duplicated(epitope_flg22$Ligand.Sequence),]
flg22_comparison <- identity_calc(epitope_flg22$Ligand.Sequence, epitope_flg22$Ligand.Sequence, "flg22")
flg22_comparison <- subset(flg22_comparison, query_id > subject_id)

# Pep-25
epitope_Pep25 <- load_training_ML_data %>% filter(Ligand == epitope_list[3])
epitope_Pep25 <- epitope_Pep25[!duplicated(epitope_Pep25$Ligand.Sequence),]
Pep25_comparison <- identity_calc(epitope_Pep25$Ligand.Sequence, epitope_Pep25$Ligand.Sequence, "Pep-25")
Pep25_comparison <- subset(Pep25_comparison, query_id > subject_id)

# SCOOP
epitope_SCOOP <- load_training_ML_data %>% filter(Ligand == epitope_list[4])
epitope_SCOOP <- epitope_SCOOP[!duplicated(epitope_SCOOP$Ligand.Sequence),]
SCOOP_comparison <- identity_calc(epitope_SCOOP$Ligand.Sequence, epitope_SCOOP$Ligand.Sequence, "SCOOP")
SCOOP_comparison <- subset(SCOOP_comparison, query_id > subject_id)

# csp22
epitope_csp22 <- load_training_ML_data %>% filter(Ligand == epitope_list[5])
epitope_csp22 <- epitope_csp22[!duplicated(epitope_csp22$Ligand.Sequence),]
csp22_comparison <- identity_calc(epitope_csp22$Ligand.Sequence, epitope_csp22$Ligand.Sequence, "csp22")
csp22_comparison <- subset(csp22_comparison, query_id > subject_id)

# elf18
epitope_elf18 <- load_training_ML_data %>% filter(Ligand == epitope_list[6])
epitope_elf18 <- epitope_elf18[!duplicated(epitope_elf18$Ligand.Sequence),]
elf18_comparison <- identity_calc(epitope_elf18$Ligand.Sequence, epitope_elf18$Ligand.Sequence, "elf18")
elf18_comparison <- subset(elf18_comparison, query_id > subject_id)

# pep/pg
epitope_peppg <- load_training_ML_data %>% filter(Ligand == epitope_list[8])
epitope_peppg <- epitope_peppg[!duplicated(epitope_peppg$Ligand.Sequence),]
peppg_comparison <- identity_calc(epitope_peppg$Ligand.Sequence, epitope_peppg$Ligand.Sequence, "pep/pg")
peppg_comparison <- subset(peppg_comparison, query_id > subject_id)

# Crip21  
epitope_crip21 <- load_training_ML_data %>% filter(Ligand == epitope_list[9])
epitope_crip21 <- epitope_crip21[!duplicated(epitope_crip21$Ligand.Sequence),]
epitope_crip21 <- subset(epitope_crip21, epitope_crip21$Ligand.Sequence != "WCRHGCCYAGSNGcIRCC")
crip21_comparison <- identity_calc(epitope_crip21$Ligand.Sequence, epitope_crip21$Ligand.Sequence, "crip21")
crip21_comparison <- subset(crip21_comparison, query_id > subject_id)

# nlp 
epitope_nlp <- load_training_ML_data %>% filter(Ligand == epitope_list[10])
epitope_nlp <- epitope_nlp[!duplicated(epitope_nlp$Ligand.Sequence),]
nlp_comparison <- identity_calc(epitope_nlp$Ligand.Sequence, epitope_nlp$Ligand.Sequence, "nlp")
nlp_comparison <- subset(nlp_comparison, query_id > subject_id)


combine_epitope_comparison <- rbind(flg22_comparison, Pep25_comparison, SCOOP_comparison, csp22_comparison, elf18_comparison, peppg_comparison, crip21_comparison, nlp_comparison)
epitope_stats <- combine_epitope_comparison %>% group_by(comparison) %>% distinct(query_id) %>% summarize(number = n())
epitope_sequence_comparison_plot <- ggplot(combine_epitope_comparison, aes(x = comparison, y = identity, fill = comparison)) +
  stat_ydensity(aes(color = comparison), alpha = 0.85, scale = "width") +
  geom_boxplot(fill = "white", width = 0.2, outlier.shape = NA) +
  theme_classic() +
  xlab("Variant Comparisons") +
  ylab("Percent Identity")+
  theme(axis.text.x = element_text(angle = 45, hjust = 1, color = "black"), 
        axis.text.y = element_text(color = "black"),
        legend.position = "none") +
  scale_y_continuous(limits = c(0, 120), breaks = c(20,40,60,80,100)) +
  coord_flip() +
  scale_fill_manual(values = epitope_colors) +
  scale_color_manual(values = epitope_colors) +
  geom_text(data = epitope_stats, aes(x = comparison, y = 110, label = number), size = 3)

ggsave(filename = "./04_Preprocessing_results/epitope_sequence_comparison_plot.pdf", plot = epitope_sequence_comparison_plot, device = "pdf", dpi = 300, width = 2.5, height = 2.5)


#load_training_ML_data$Ligand.Length <- as.numeric(nchar(load_training_ML_data$Ligand.Sequence))
# plot the peptide length distribution
#ggplot(load_training_ML_data, aes(x = Ligand, y = Ligand.Length)) +
#see::geom_violinhalf() +
#  theme_classic(scale = "width") +
#  xlab("Peptide") +
 # ylab("Length")


# ridge plot
#ggplot(load_training_ML_data, aes(x = Ligand, y = Ligand.Length)) +
 # ggridges::geom_density_ridges(scale = 2) +
  #xlab("Peptide") +
  #ylab("Length") +
  #scale_y_discrete(expand = c(0, 0)) +     # will generally have to set the `expand` option
  #scale_x_continuous(expand = c(0, 0)) +   # for both axes to remove unneeded padding
  #coord_cartesian(clip = "off") + # to avoid clipping of the very top of the top ridgeline
  #theme_ridges()



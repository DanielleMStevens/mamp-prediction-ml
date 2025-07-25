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
library(Biostrings, warn.conflicts = FALSE, quietly = TRUE)
library(pwalign, warn.conflicts = FALSE, quietly = TRUE)
library(ggplot2, warn.conflicts = FALSE, quietly = TRUE)
library(ggridges, warn.conflicts = FALSE, quietly = TRUE)

# color code for genera of interest
epitope_colors <- c("#b35c46","#e2b048","#ebd56d","#b9d090","#37a170","#86c0ce","#7d9fc6", "#32527B", "#542a64", "#232232","#D5869D")
names(epitope_colors) <- c("crip21","csp22","elf18","flg22","flgII-28","In11","nlp", "pep-25", "pg", "scoop","screw")

receptor_colors <- c("#b35c46","#e2b048","#ebd56d","#b9d090","#37a170","#86c0ce","#7d9fc6", "#32527B", "#542a64", "#232232","#D5869D")
names(receptor_colors) <- c("CuRe1","CORE","EFR","FLS2","FLS3","INR","RLP23", "PERU", "RLP42", "MIK2","NUT")

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
#  plot of all input data by peptide immunogenicity
######################################################################

# distribution of peptide outcomes
peptide_distrubution <- load_training_ML_data %>% group_by(Ligand, Immunogenicity) %>% summarize(n=n())
immunogenicity_distrubution <- ggplot(data = peptide_distrubution, aes(x=Immunogenicity, y=n, fill=Ligand)) +
  geom_bar(stat="identity") +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8, color = "black"),
        axis.text.y = element_text(color = "black"),
        legend.position = "none")+
  labs(x="", y="Count") +
  scale_fill_manual(values = epitope_colors) +
  geom_text(data = peptide_distrubution %>% 
              group_by(Immunogenicity) %>% 
              summarise(n = sum(n)),
            aes(label = n, y = n, x = Immunogenicity), 
            position = position_stack(vjust = 1.05),
            inherit.aes = FALSE,
            size = 3)

ggsave(filename = "./04_Preprocessing_results/peptide_distrubution.pdf", plot = immunogenicity_distrubution, device = "pdf", dpi = 300, width = 1.5, height = 3)


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

# rlp42 receptor
receptor_RLP42 <- load_training_ML_data %>% filter(Receptor == receptors_list[9])
receptor_RLP42 <- receptor_RLP42[!duplicated(receptor_RLP42$Receptor.Sequence),]
RLP42_comparison <- identity_calc(receptor_RLP42$Locus.ID.Genbank, receptor_RLP42$Receptor.Sequence, "RLP42")
RLP42_comparison <- subset(RLP42_comparison, query_id != subject_id)

# rlp23 receptor
receptor_RLP23 <- load_training_ML_data %>% filter(Receptor == receptors_list[11])
receptor_RLP23 <- receptor_RLP23[!duplicated(receptor_RLP23$Receptor.Sequence),]
RLP23_comparison <- identity_calc(receptor_RLP23$Locus.ID.Genbank, receptor_RLP23$Receptor.Sequence, "RLP23")
RLP23_comparison <- subset(RLP23_comparison, query_id != subject_id)

# nut receptor
receptor_NUT <- load_training_ML_data %>% filter(Receptor == receptors_list[12])
receptor_NUT <- receptor_NUT[!duplicated(receptor_NUT$Receptor.Sequence),]
NUT_comparison <- identity_calc(receptor_NUT$Locus.ID.Genbank, receptor_NUT$Receptor.Sequence, "NUT")
NUT_comparison <- subset(NUT_comparison, query_id != subject_id)

combine_receptor_comparison <- rbind(FLS2_comparison, PERU_comparison, MIK2_comparison, CORE_comparison, EFR_comparison, 
                                    FLS3_comparison, INR_comparison, RLP42_comparison, RLP23_comparison, NUT_comparison)
receptor_stats <- combine_receptor_comparison %>% group_by(comparison) %>% distinct(query_id) %>% summarize(number = n())
receptor_stats$number <- receptor_stats$number + 1

receptor_sequence_comparison_plot <- ggplot(combine_receptor_comparison, aes(x = comparison, y = identity, fill = comparison)) +
  stat_ydensity(aes(color = comparison), alpha = 0.85, scale = "width") +
  geom_boxplot(fill = "white", width = 0.25, outlier.shape = NA) +
  theme_classic() +
  xlab("Homolog Comparisons") +
  ylab("Percent Identity")+
  theme(axis.text.x = element_text(angle = 45, hjust = 1, color = "black"), 
        axis.text.y = element_text(color = "black"),
        legend.position = "none") +
  scale_y_continuous(limits = c(0, 120), breaks = c(20,40,60,80,100)) +
  coord_flip() +
  scale_fill_manual(values = receptor_colors) +
  scale_color_manual(values = receptor_colors) +
  geom_text(data = receptor_stats, aes(x = comparison, y = 110, label = number), size = 3)

ggsave(filename = "./04_Preprocessing_results/receptor_sequence_comparison_plot.pdf", 
plot = receptor_sequence_comparison_plot, device = "pdf", dpi = 300, width = 2.5, height = 2.35)


######################################################################
#  parse and compare epitope variants
######################################################################

# subset training data into fasta files based on receptor/ligand
epitope_list <- unique(load_training_ML_data$Ligand)

# In11 epitope
epitope_In11 <- load_training_ML_data %>% filter(Ligand == epitope_list[1])
epitope_In11 <- epitope_In11[!duplicated(epitope_In11$Ligand.Sequence),]
In11_comparison <- identity_calc(epitope_In11$Ligand.Sequence, epitope_In11$Ligand.Sequence, "In11")
In11_comparison <- subset(In11_comparison, query_id > subject_id)

# flg22 epitope
epitope_flg22 <- load_training_ML_data %>% filter(Ligand == epitope_list[2])
epitope_flg22 <- epitope_flg22[!duplicated(epitope_flg22$Ligand.Sequence),]
flg22_comparison <- identity_calc(epitope_flg22$Ligand.Sequence, epitope_flg22$Ligand.Sequence, "flg22")
flg22_comparison <- subset(flg22_comparison, query_id > subject_id)

# Pep-25
epitope_Pep25 <- load_training_ML_data %>% filter(Ligand == epitope_list[3])
epitope_Pep25 <- epitope_Pep25[!duplicated(epitope_Pep25$Ligand.Sequence),]
Pep25_comparison <- identity_calc(epitope_Pep25$Ligand.Sequence, epitope_Pep25$Ligand.Sequence, "pep-25")
Pep25_comparison <- subset(Pep25_comparison, query_id > subject_id)

# SCOOP
epitope_SCOOP <- load_training_ML_data %>% filter(Ligand == epitope_list[4])
epitope_SCOOP <- epitope_SCOOP[!duplicated(epitope_SCOOP$Ligand.Sequence),]
SCOOP_comparison <- identity_calc(epitope_SCOOP$Ligand.Sequence, epitope_SCOOP$Ligand.Sequence, "scoop")
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

# flgII-28
epitope_flgII28 <- load_training_ML_data %>% filter(Ligand == epitope_list[7])
epitope_flgII28 <- epitope_flgII28[!duplicated(epitope_flgII28$Ligand.Sequence),]
flgII28_comparison <- identity_calc(epitope_flgII28$Ligand.Sequence, epitope_flgII28$Ligand.Sequence, "flgII-28")
flgII28_comparison <- subset(flgII28_comparison, query_id > subject_id)

# pg
epitope_peppg <- load_training_ML_data %>% filter(Ligand == epitope_list[8])
epitope_peppg <- epitope_peppg[!duplicated(epitope_peppg$Ligand.Sequence),]
peppg_comparison <- identity_calc(epitope_peppg$Ligand.Sequence, epitope_peppg$Ligand.Sequence, "pg")
peppg_comparison <- subset(peppg_comparison, query_id > subject_id)

# crip21  
epitope_crip21 <- load_training_ML_data %>% filter(Ligand == epitope_list[9])
epitope_crip21 <- epitope_crip21[!duplicated(epitope_crip21$Ligand.Sequence),]
crip21_comparison <- identity_calc(epitope_crip21$Ligand.Sequence, epitope_crip21$Ligand.Sequence, "crip21")
crip21_comparison <- subset(crip21_comparison, query_id > subject_id)

# nlp 
epitope_nlp <- load_training_ML_data %>% filter(Ligand == epitope_list[10])
epitope_nlp <- epitope_nlp[!duplicated(epitope_nlp$Ligand.Sequence),]
nlp_comparison <- identity_calc(epitope_nlp$Ligand.Sequence, epitope_nlp$Ligand.Sequence, "nlp")
nlp_comparison <- subset(nlp_comparison, query_id > subject_id)

# screw
epitope_screw <- load_training_ML_data %>% filter(Ligand == epitope_list[11])
epitope_screw <- epitope_screw[!duplicated(epitope_screw$Ligand.Sequence),]
screw_comparison <- identity_calc(epitope_screw$Ligand.Sequence, epitope_screw$Ligand.Sequence, "screw")
screw_comparison <- subset(screw_comparison, query_id > subject_id)

combine_epitope_comparison <- rbind(In11_comparison, flg22_comparison, Pep25_comparison, SCOOP_comparison, csp22_comparison, 
                                    elf18_comparison, flgII28_comparison, peppg_comparison, crip21_comparison, nlp_comparison, screw_comparison)
epitope_stats <- combine_epitope_comparison %>% group_by(comparison) %>% distinct(query_id) %>% summarize(number = n())
epitope_stats$number <- epitope_stats$number + 1

epitope_sequence_comparison_plot <- ggplot(combine_epitope_comparison, aes(x = comparison, y = identity, fill = comparison)) +
  stat_ydensity(aes(color = comparison), alpha = 0.85, scale = "width") +
  geom_boxplot(fill = "white", width = 0.25, outlier.shape = NA) +
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

ggsave(filename = "./04_Preprocessing_results/epitope_sequence_comparison_plot.pdf", 
plot = epitope_sequence_comparison_plot, device = "pdf", dpi = 300, width = 2.5, height = 2.55)


######################################################################
#  parse and compare epitope variant lengths
######################################################################

# ridge plot
epitope_length_comparison_plot <- ggplot(load_training_ML_data, aes(x = Ligand.Length, y = Ligand)) +
  ggridges::geom_density_ridges(aes(fill = Ligand), 
      rel_min_height = 0.01, alpha = 0.85, scale = 1.5) +
  xlab("Length") +
  ylab("") +
  scale_fill_manual(values = epitope_colors) +
  scale_y_discrete(expand = c(0, 0)) +     # will generally have to set the `expand` option
  scale_x_continuous(expand = c(0, 0)) +   # for both axes to remove unneeded padding
  theme_ridges(center = TRUE, font_size = 8) +
  theme(legend.position = "none")

ggsave(filename = "./04_Preprocessing_results/epitope_length_comparison_plot.pdf", 
plot = epitope_length_comparison_plot, device = "pdf", dpi = 300, width = 2, height = 2)

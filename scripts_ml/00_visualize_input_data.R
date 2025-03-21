#-----------------------------------------------------------------------------------------------
# Coaker Lab - Plant Pathology Department UC Davis
# Author: Danielle M. Stevens
# Last Updated: 07/06/2020
# Script Purpose: 
# Inputs: 
# Outputs: 
#-----------------------------------------------------------------------------------------------

library(readxl)
library(tidyverse)
library(webr)
library(networkD3)
library(Biostrings)
library(pwalign)

load_training_ML_data <- readxl::read_xlsx(path = "./in_data/All_LRR_PRR_ligand_data.xlsx")
load_training_ML_data <- data.frame(load_training_ML_data)[1:12]


# distribution of peptide outcomes
peptide_distrubution <- load_training_ML_data %>% group_by(Ligand, Immunogenicity) %>% summarize(n=n())
webr::PieDonut(peptide_distrubution, aes(Immunogenicity,Ligand, count=n))

library(paletteer)
ggplot(peptide_distrubution, aes(x=Immunogenicity, y=n, fill=Ligand)) +
  geom_bar(stat="identity") +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size =8))+
  labs(x="",
       y="Count") +
  scale_fill_paletteer_d("MexBrewer::Alacena")


# compare sequences of eptiopes and receptors to document sequence variation
# Install Bioconductor
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install()

# Install package dependencies
BiocManager::install(c(
        "Biostrings",
        "GenomicRanges",
        "GenomicFeatures",
        "Rsamtools",
        "rtracklayer"
))

# install CRAN dependencies
install.packages(c("doParallel", "foreach", "ape", "Rdpack", "benchmarkme", "devtools"))

# install BLAST dependency metablastr from GitHub
devtools::install_github("drostlab/metablastr")

# install DIAMOND dependency rdiamond from GitHub
devtools::install_github("drostlab/rdiamond")

# install orthologr from GitHub
devtools::install_github("drostlab/orthologr")

library(orthologr)

#writeFasta(formate2fasta(receptor_INR$Locus.ID.Genbank, "INR", receptor_INR$Receptor.Sequence), "./out_data/training_data_summary/receptor_INR.fasta")
#writeFasta(formate2fasta(receptor_FLS2$Locus.ID.Genbank, "FLS2", receptor_FLS2$Receptor.Sequence), "./out_data/training_data_summary/receptor_FLS2.fasta")
#writeFasta(formate2fasta(receptor_PERU$Locus.ID.Genbank, "PERU", receptor_PERU$Receptor.Sequence), "./out_data/training_data_summary/receptor_PERU.fasta")
#writeFasta(formate2fasta(receptor_MIK2$Locus.ID.Genbank, "MIK2", receptor_MIK2$Receptor.Sequence), "./out_data/training_data_summary/receptor_MIK2.fasta")
#writeFasta(formate2fasta(receptor_CORE$Locus.ID.Genbank, "CORE", receptor_CORE$Receptor.Sequence), "./out_data/training_data_summary/receptor_CORE.fasta")
#writeFasta(formate2fasta(receptor_EFR$Locus.ID.Genbank, "EFR", receptor_EFR$Receptor.Sequence), "./out_data/training_data_summary/receptor_EFR.fasta")
#writeFasta(formate2fasta(receptor_FLS3$Locus.ID.Genbank, "FLS3", receptor_FLS3$Receptor.Sequence), "./out_data/training_data_summary/receptor_FLS3.fasta")

#INR_comparison <- as.data.frame(orthologr::blast(query_file = "./out_data/training_data_summary/receptor_INR.fasta",
#                                              subject_file = "./out_data/training_data_summary/receptor_INR.fasta", 
#                                              seq_type = 'protein'))


# convert training/test data to fasta format
######################################################################
#  function to turn dataframe (where one column is the name and one column is the sequence) into a fasta file - version 2
######################################################################

formate2fasta <- function(WP_locus_names, sequence_type, sequences) {
  hold_sequences <- data.frame("Locus_Tag_Name" = character(0), "Sequence" = character(0))
  pb <- txtProgressBar(min = 0, max = length(WP_locus_names), style = 3)

  for (i in 1:length(WP_locus_names)){
    #find full length protein sequence
    temp_df <- data.frame(paste(paste(">",i, sep=""), sequence_type, WP_locus_names[[i]], sep = "|"),
                          sequences[[i]])
    colnames(temp_df) <- colnames(hold_sequences)
    hold_sequences <- rbind(hold_sequences, temp_df)
    setTxtProgressBar(pb, i)
  }
  close(pb)
  return(hold_sequences)
}


######################################################################
#  function to turn dataframe (where one column is the name and one column is the sequence) into a fasta file
######################################################################

writeFasta <- function(data, filename){
  fastaLines = c()
  for (rowNum in 1:nrow(data)){
    fastaLines = c(fastaLines, data[rowNum,1])
    fastaLines = c(fastaLines,data[rowNum,2])
  }
  fileConn<-file(filename)
  writeLines(fastaLines, fileConn)
  close(fileConn)
}


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
ggplot(combine_receptor_comparison, aes(x = comparison, y = identity, fill = comparison)) +
  stat_ydensity(aes(color = comparison), alpha = 0.85, scale = "width") +
  geom_boxplot(fill = "white", width = 0.2, outlier.shape = NA) +
  theme_classic() +
  xlab("Homolog Comparisons") +
  ylab("Percent Identity")+
  theme(axis.text.x = element_text(angle = 45, hjust = 1, color = "black"), axis.text.y = element_text(color = "black")) +
  scale_y_continuous(limits = c(20, 120), breaks = c(20,40,60,80,100)) +
  coord_flip() +
  scale_fill_paletteer_d("NatParksPalettes::Banff") +
  scale_color_paletteer_d("NatParksPalettes::Banff") +
  geom_text(data = receptor_stats, aes(x = comparison, y = 110, label = number), size = 3)



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
crip21_comparison <- identity_calc(epitope_crip21$Ligand.Sequence, epitope_crip21$Ligand.Sequence, "crip21")
crip21_comparison <- subset(crip21_comparison, query_id > subject_id)

# nlp 
epitope_nlp <- load_training_ML_data %>% filter(Ligand == epitope_list[10])
epitope_nlp <- epitope_nlp[!duplicated(epitope_nlp$Ligand.Sequence),]
nlp_comparison <- identity_calc(epitope_nlp$Ligand.Sequence, epitope_nlp$Ligand.Sequence, "nlp")
nlp_comparison <- subset(nlp_comparison, query_id > subject_id)


combine_epitope_comparison <- rbind(flg22_comparison, Pep25_comparison, SCOOP_comparison, csp22_comparison, elf18_comparison, peppg_comparison, nlp_comparison)
epitope_stats <- combine_epitope_comparison %>% group_by(comparison) %>% distinct(query_id) %>% summarize(number = n())
ggplot(combine_epitope_comparison, aes(x = comparison, y = identity, fill = comparison)) +
  stat_ydensity(aes(color = comparison), alpha = 0.85, scale = "width") +
  geom_boxplot(fill = "white", width = 0.2, outlier.shape = NA) +
  theme_classic() +
  xlab("Homolog Comparisons") +
  ylab("Percent Identity")+
  theme(axis.text.x = element_text(angle = 45, hjust = 1, color = "black"), axis.text.y = element_text(color = "black")) +
  scale_y_continuous(limits = c(20, 120), breaks = c(20,40,60,80,100)) +
  coord_flip() +
  scale_fill_paletteer_d("NatParksPalettes::Banff") +
  scale_color_paletteer_d("NatParksPalettes::Banff") +
  geom_text(data = epitope_stats, aes(x = comparison, y = 110, label = number), size = 3)






#-----------------------------------------------------------------------------------------------
# Prepare data for Sankey diagram
# Create nodes dataframe
nodes <- data.frame(
  name = c(unique(load_training_ML_data$Receptor),  unique(load_training_ML_data$Immunogenicity), unique(load_training_ML_data$Ligand)))

# Create links dataframe with color groups
links_receptor_immuno <- load_training_ML_data %>%
  group_by(Receptor, Immunogenicity) %>%
  summarise(value = n()) %>%
  mutate(
    source = match(Receptor, nodes$name) - 1, 
    target = match(Immunogenicity, nodes$name) - 1,
    group = Immunogenicity
  )

links_immuno_ligand <- load_training_ML_data %>%
  group_by(Immunogenicity, Ligand) %>%
  summarise(value = n()) %>%
  mutate(
    source = match(Immunogenicity, nodes$name) - 1, 
    target = match(Ligand, nodes$name) - 1,
    group = Immunogenicity
  )

# Combine all links
links <- rbind(links_receptor_immuno, links_immuno_ligand)

# Define color scheme for immunogenicity with alpha = 0.7
my_color <- JS('d3.scaleOrdinal()
  .domain(["Immunogenic", "Weakly Immunogenic", "Non-Immunogenic"])
  .range(["rgba(64, 64, 64, 0.7)", "rgba(0, 0, 139, 0.7)", "rgba(139, 0, 0, 0.7)"])')

# Create and display Sankey diagram
sankeyNetwork(Links = links, Nodes = nodes, Source = "source",
             Target = "target", Value = "value", NodeID = "name",
             sinksRight = TRUE, nodeWidth = 30, fontSize = 12, height = 500,
             LinkGroup = "group", colourScale = my_color,
             iterations = 0,  # Helps with layout stability
             nodePadding = 20)  # Adds more space between nodes


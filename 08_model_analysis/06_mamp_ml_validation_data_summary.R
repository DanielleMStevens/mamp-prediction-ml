# 06 analyze mamp-ml validation dataset predictions

# Load libraries
library(dplyr, warn.conflicts = FALSE, quietly = TRUE)
library(readr, warn.conflicts = FALSE, quietly = TRUE)
library(Biostrings, warn.conflicts = FALSE, quietly = TRUE)
library(pwalign, warn.conflicts = FALSE, quietly = TRUE)
library(ggplot2, warn.conflicts = FALSE, quietly = TRUE)
library(ggridges, warn.conflicts = FALSE, quietly = TRUE)


# color code for genera of interest
epitope_colors <- c("#b35c46","#e2b048","#ebd56d","#b9d090","#37a170","#86c0ce","#7d9fc6", "#32527B", "#542a64", "#232232","#D5869D")
names(epitope_colors) <- c("crip21","csp22","elf18","flg22","flgII-28","In11","nlp", "pep-25", "pg", "scoop","screw")

receptor_colors <- c("#b35c46","#e2b048","#ebd56d","#b9d090","#37a170","#86c0ce", "#86c0ce","#7d9fc6", "#32527B", "#542a64", "#232232","#D5869D")
names(receptor_colors) <- c("CuRe1","CORE","EFR","FLS2","FLS3","INR", "INR-like","RLP23", "PERU", "RLP42", "MIK2","NUT")


# Define file paths
correct_file <- "07_model_results/00_mamp_ml/correct_classification_report.tsv"
misclassified_file <- "07_model_results/00_mamp_ml/misclassification_report.tsv"

# Read the data
correct_data <- read_tsv(correct_file, show_col_types = FALSE)
misclassified_data <- read_tsv(misclassified_file, show_col_types = FALSE)

# Add classification status
correct_data <- correct_data %>%
  mutate(ClassificationStatus = "Correct")

misclassified_data <- misclassified_data %>%
  mutate(ClassificationStatus = "Misclassified")

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

# lets only analyze the top three receptor-epitope combinations (FLS2, MIK2, and CORE) 
sequence_analysis_receptor <- function(data_frame_in, receptor_in, classification_status_in){
    data_frame_in <- subset(data_frame_in, data_frame_in$Receptor == receptor_in)
    data_frame_in <- data_frame_in %>% distinct(Receptor_sequence, .keep_all = TRUE)
    comparison <- identity_calc(paste(data_frame_in$Plant_species, data_frame_in$Receptor, sep= "_"), data_frame_in$Receptor_sequence, receptor_in)
    comparison <- subset(comparison, identity != 100.00000)
    comparison$ClassificationStatus <- classification_status_in
    return(comparison)
}

FLS2_misclassified <- sequence_analysis_receptor(misclassified_data, "FLS2", "Misclassified")
FLS2_correct <- sequence_analysis_receptor(correct_data, "FLS2", "Correct")
MIK2_correct <- sequence_analysis_receptor(correct_data, "MIK2", "Correct")
CORE_correct <- sequence_analysis_receptor(correct_data, "CORE", "Correct")

receptors_combined <- rbind(FLS2_misclassified, FLS2_correct, MIK2_correct, CORE_correct)

receptors_combined_comparison <- ggplot(receptors_combined, aes(x = factor(comparison, levels = c("CORE", "MIK2", "FLS2")), y = identity, fill = ClassificationStatus)) +
    stat_ydensity(alpha = 0.85, scale = "width") +
    geom_boxplot( fill = "white", width = 0.25, outlier.shape = NA) +
    theme_classic() +
    labs(x = "Receptor", y = "Percent Identity") +
    theme(legend.position = "bottom",
          legend.direction = "vertical",
          panel.grid.major = element_line(color = "grey90"),
          panel.grid.minor = element_line(color = "grey95"),
          axis.text = element_text(color = "black", size = 9),
          axis.title = element_text(color = "black", size = 10), 
          legend.text = element_text(color = "black", size = 6),
          legend.title = element_text(color = "black", size = 6),
          legend.key.size = unit(0.3, "cm"),
          legend.spacing = unit(0.1, "cm")) +
    scale_y_continuous(limits = c(0, 120), breaks = c(20,40,60,80,100)) +
    coord_flip() +
    facet_wrap(~ClassificationStatus, dir = "v") +
    scale_fill_manual(values = c("Correct" = "#97BC62", "Misclassified" = "#2C5F2D"))

ggsave("08_model_analysis/mamp_ml_validation_data_receptor_summary.pdf", receptors_combined_comparison, width = 2.3, height = 3.6, dpi = 300)

# lets only analyze the top three receptor-epitope combinations (flg22, scoop, and csp22) 
sequence_analysis_epitope <- function(data_frame_in, epitope_in, classification_status_in){
    data_frame_in <- subset(data_frame_in, data_frame_in$Ligand == epitope_in)
    data_frame_in <- data_frame_in %>% distinct(Ligand_sequence, .keep_all = TRUE)
    comparison <- identity_calc(paste(data_frame_in$Ligand, base::seq_along(data_frame_in$Ligand), sep = "_"), data_frame_in$Ligand_sequence, epitope_in)
    comparison <- subset(comparison, identity != 100.00000)
    comparison$ClassificationStatus <- classification_status_in
    return(comparison)
}

flg22_misclassified <- sequence_analysis_epitope(misclassified_data, "flg22", "Misclassified")
flg22_correct <- sequence_analysis_epitope(correct_data, "flg22", "Correct")

scoop_misclassified <- sequence_analysis_epitope(misclassified_data, "scoop", "Misclassified")
scoop_correct <- sequence_analysis_epitope(correct_data, "scoop", "Correct")

csp22_misclassified <- sequence_analysis_epitope(misclassified_data, "csp22", "Misclassified")
csp22_correct <- sequence_analysis_epitope(correct_data, "csp22", "Correct")

epitope_combined <- rbind(flg22_misclassified, flg22_correct, scoop_misclassified, scoop_correct, csp22_misclassified, csp22_correct)

epitope_combined_comparison <- ggplot(epitope_combined, aes(x = factor(comparison, levels = c("csp22", "scoop", "flg22")), y = identity, fill = ClassificationStatus)) +
    stat_ydensity(alpha = 0.85, scale = "width") +
    geom_boxplot( fill = "white", width = 0.25, outlier.shape = NA) +
    theme_classic() +
    labs(x = "Epitope", y = "Percent Identity") +
    theme(legend.position = "bottom",
          legend.direction = "vertical",
          panel.grid.major = element_line(color = "grey90"),
          panel.grid.minor = element_line(color = "grey95"),
          axis.text = element_text(color = "black", size = 9),
          axis.title = element_text(color = "black", size = 10), 
          legend.text = element_text(color = "black", size = 6),
          legend.title = element_text(color = "black", size = 6),
          legend.key.size = unit(0.3, "cm"),
          legend.spacing = unit(0.1, "cm")) +
    scale_y_continuous(limits = c(0, 120), breaks = c(20,40,60,80,100)) +
    coord_flip() +
    facet_wrap(~ClassificationStatus, dir = "v") +
    scale_fill_manual(values = c("Correct" = "#97BC62", "Misclassified" = "#2C5F2D"))

ggsave("08_model_analysis/mamp_ml_validation_data_epitope_summary.pdf", epitope_combined_comparison, width = 2.3, height = 3.6, dpi = 300)


# ----------------------- analyze epitope length using ggridges -----------------------

correct_data$ligand_length <- nchar(correct_data$Ligand_sequence)
misclassified_data$ligand_length <- nchar(misclassified_data$Ligand_sequence)

correct_predictions_ligand_plot <-ggplot(correct_data[correct_data$Ligand %in% c("csp22","flg22","scoop"),], 
                aes(x = ligand_length, y = factor(Ligand, levels = c("csp22", "scoop", "flg22")))) + 
    geom_density_ridges(fill = "#97BC62", alpha = 0.85) +
    theme_ridges() +
    labs(x = "Epitope Length", y = "Epitope") +
    theme(legend.position = "bottom",
          legend.direction = "vertical",
          panel.grid.major = element_line(color = "grey90"),
          panel.grid.minor = element_line(color = "grey95"),
          axis.text = element_text(color = "black", size = 9),
          axis.title = element_text(color = "black", size = 10))

ggsave("08_model_analysis/correct_predictions_ligand_plot.pdf", correct_predictions_ligand_plot, width = 1.8, height = 1.6, dpi = 300)


misclassified_predictions_ligand_plot <- ggplot(misclassified_data[misclassified_data$Ligand %in% c("csp22","flg22","scoop"),],
                aes(x = ligand_length, y = factor(Ligand, levels = c("csp22", "scoop", "flg22")))) + 
geom_density_ridges(fill = "#2C5F2D", alpha = 0.85) +
    theme_ridges() +
    labs(x = "Epitope Length", y = "Epitope") +
    theme(legend.position = "bottom",
          legend.direction = "vertical",
          panel.grid.major = element_line(color = "grey90"),
          panel.grid.minor = element_line(color = "grey95"),
          axis.text = element_text(color = "black", size = 9),
          axis.title = element_text(color = "black", size = 10))

ggsave("08_model_analysis/misclassified_predictions_ligand_plot.pdf", misclassified_predictions_ligand_plot, width = 1.8, height = 1.6, dpi = 300)




# export the ligand sequence into a fasta file to make weblogos with
# ---------------------------- flg22 eptiope sequence analysis ----------------------------
flg22_correct_fasta <- correct_data[correct_data$Ligand == "flg22",]
flg22_correct_fasta <- data.frame(Header_Name = paste(">",flg22_correct_fasta$Ligand, base::seq_along(flg22_correct_fasta$Ligand), sep = "_"), 
                                  Receptor_Sequence = flg22_correct_fasta$Ligand_sequence)
flg22_correct_fasta <- flg22_correct_fasta %>% distinct(Receptor_Sequence, .keep_all = TRUE)

# flg22 correct prediction sequences
write.table(flg22_correct_fasta, file = "./08_model_analysis/Epitope_sequence_analysis/flg22_correct_prediction_sequences.fasta", 
            quote = FALSE, row.names = FALSE, col.names = FALSE, sep = "\n")


flg22_correct_fasta <- misclassified_data[misclassified_data$Ligand == "flg22",]
flg22_correct_fasta <- data.frame(Header_Name = paste(">",flg22_correct_fasta$Ligand, base::seq_along(flg22_correct_fasta$Ligand), sep = "_"), 
                                  Receptor_Sequence = flg22_correct_fasta$Ligand_sequence)
flg22_correct_fasta <- flg22_correct_fasta %>% distinct(Receptor_Sequence, .keep_all = TRUE)

# flg22 misclassified prediction sequences
write.table(flg22_correct_fasta, file = "./08_model_analysis/Epitope_sequence_analysis/flg22_misclassified_prediction_sequences.fasta", 
            quote = FALSE, row.names = FALSE, col.names = FALSE, sep = "\n")


# ---------------------------- scoop eptiope sequence analysis ----------------------------

scoop_correct_fasta <- correct_data[correct_data$Ligand == "scoop",]
scoop_correct_fasta <- data.frame(Header_Name = paste(">",scoop_correct_fasta$Ligand, base::seq_along(scoop_correct_fasta$Ligand), sep = "_"), 
                                  Receptor_Sequence = scoop_correct_fasta$Ligand_sequence)
scoop_correct_fasta <- scoop_correct_fasta %>% distinct(Receptor_Sequence, .keep_all = TRUE)

# scoop correct prediction sequences
write.table(scoop_correct_fasta, file = "./08_model_analysis/Epitope_sequence_analysis/scoop_correct_prediction_sequences.fasta", 
            quote = FALSE, row.names = FALSE, col.names = FALSE, sep = "\n")


scoop_correct_fasta <- misclassified_data[misclassified_data$Ligand == "scoop",]
scoop_correct_fasta <- data.frame(Header_Name = paste(">",scoop_correct_fasta$Ligand, base::seq_along(scoop_correct_fasta$Ligand), sep = "_"), 
                                  Receptor_Sequence = scoop_correct_fasta$Ligand_sequence)
scoop_correct_fasta <- scoop_correct_fasta %>% distinct(Receptor_Sequence, .keep_all = TRUE)

# scoop misclassified prediction sequences
write.table(scoop_correct_fasta, file = "./08_model_analysis/Epitope_sequence_analysis/scoop_misclassified_prediction_sequences.fasta", 
            quote = FALSE, row.names = FALSE, col.names = FALSE, sep = "\n")

# ---------------------------- csp22 eptiope sequence analysis ----------------------------

csp22_correct_fasta <- correct_data[correct_data$Ligand == "csp22",]
csp22_correct_fasta <- data.frame(Header_Name = paste(">",csp22_correct_fasta$Ligand, base::seq_along(csp22_correct_fasta$Ligand), sep = "_"), 
                                  Receptor_Sequence = csp22_correct_fasta$Ligand_sequence)
csp22_correct_fasta <- csp22_correct_fasta %>% distinct(Receptor_Sequence, .keep_all = TRUE)

# csp22 correct prediction sequences
write.table(csp22_correct_fasta, file = "./08_model_analysis/Epitope_sequence_analysis/csp22_correct_prediction_sequences.fasta", 
            quote = FALSE, row.names = FALSE, col.names = FALSE, sep = "\n")


csp22_correct_fasta <- misclassified_data[misclassified_data$Ligand == "csp22",]
csp22_correct_fasta <- data.frame(Header_Name = paste(">",csp22_correct_fasta$Ligand, base::seq_along(csp22_correct_fasta$Ligand), sep = "_"), 
                                  Receptor_Sequence = csp22_correct_fasta$Ligand_sequence)
csp22_correct_fasta <- csp22_correct_fasta %>% distinct(Receptor_Sequence, .keep_all = TRUE)

# csp22 misclassified prediction sequences
write.table(csp22_correct_fasta, file = "./08_model_analysis/Epitope_sequence_analysis/csp22_misclassified_prediction_sequences.fasta", 
            quote = FALSE, row.names = FALSE, col.names = FALSE, sep = "\n")
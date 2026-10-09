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
library(showtext, warn.conflicts = FALSE, quietly = TRUE)
library(sysfonts, warn.conflicts = FALSE, quietly = TRUE)

# Register system fonts for correct PDF glyph metrics (fixes spaced-out letters).
# Use real TTF for Arial; Helvetica Neue from the system collection.
font_add(
  "Arial",
  regular = "/System/Library/Fonts/Supplemental/Arial.ttf",
  bold = "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
  italic = "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
  bolditalic = "/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf"
)
font_add("Helvetica Neue", regular = "/System/Library/Fonts/HelveticaNeue.ttc")
showtext_auto()
showtext_opts(dpi = 300)

# color code for genera of interest
epitope_colors <- c("#b35c46","#e2b048","#ebd56d","#b9d090","#37a170","#86c0ce","#7d9fc6", "#32527B", "#542a64", "#232232","#D5869D", "#db7407")
names(epitope_colors) <- c("crip21","csp22","elf18","flg22","flgII-28","In11","nlp", "pep-25", "pg", "scoop","screw", "phytocytokine")

receptor_colors <- c("#b35c46","#e2b048","#ebd56d","#b9d090","#37a170","#86c0ce","#c5b3d0","#c5b3d0","#7d9fc6", "#32527B", "#542a64", "#232232","#D5869D", "#db7407","#7a1840","#1A7A6D","#b3adad")
names(receptor_colors) <- c("CuRe1","CORE","EFR","FLS2","FLS3","INR","LNR","LNR-like","RLP23", "PERU", "RLP42", "MIK2","NUT", "phytocytokine","PEPR","SCORE","unknown")

# Thin, clean journal aesthetic (L-frame, short ticks, no clutter)
theme_clean_thin <- function(base_size = 8) {
  theme_classic(base_size = base_size, base_family = "Arial") %+replace%
    theme(
      line = element_line(colour = "black", linewidth = 0.3),
      rect = element_rect(fill = "white", colour = NA, linewidth = 0.3),
      text = element_text(colour = "black", family = "Arial"),
      axis.line = element_line(colour = "black", linewidth = 0.35),
      axis.ticks = element_line(colour = "black", linewidth = 0.3),
      axis.ticks.length = unit(1.2, "pt"),
      axis.title = element_text(size = base_size, colour = "black", family = "Arial"),
      axis.text = element_text(size = base_size - 1, colour = "black", family = "Arial"),
      panel.background = element_rect(fill = "white", colour = NA),
      panel.grid = element_blank(),
      plot.background = element_rect(fill = "white", colour = NA),
      legend.background = element_blank(),
      legend.key = element_blank(),
      legend.position = "none",
      strip.background = element_blank()
    )
}

# Load all workbook tabs and keep the shared first 12 columns
excel_path <- "./02_in_data/All_LRR_PRR_ligand_data.xlsx"
sheet_names <- readxl::excel_sheets(excel_path)
load_training_ML_data <- dplyr::bind_rows(lapply(sheet_names, function(sheet) {
  df <- data.frame(readxl::read_xlsx(path = excel_path, sheet = sheet), stringsAsFactors = FALSE)
  df <- df[, 1:12, drop = FALSE]
  # Tabs differ in column types (e.g. Number of LRRs); coerce for bind_rows
  df[] <- lapply(df, as.character)
  df$Source_Sheet <- sheet
  df
}))
# Restore numeric length after bind
load_training_ML_data$Ligand.Length <- suppressWarnings(as.numeric(load_training_ML_data$Ligand.Length))

# Harmonize labels that differ across tabs
load_training_ML_data <- load_training_ML_data %>%
  dplyr::mutate(
    Receptor = dplyr::case_when(
      tolower(as.character(Receptor)) == "unknown" ~ "unknown",
      TRUE ~ as.character(Receptor)
    ),
    Immunogenicity = dplyr::case_when(
      tolower(as.character(Immunogenicity)) == "immunogenic" ~ "Immunogenic",
      tolower(as.character(Immunogenicity)) == "non-immunogenic" ~ "Non-Immunogenic",
      tolower(as.character(Immunogenicity)) == "weakly immunogenic" ~ "Weakly Immunogenic",
      TRUE ~ as.character(Immunogenicity)
    )
  )


######################################################################
#  function to calculate 
######################################################################

identity_calc <- function(label_ids, sequence_list, comparison){
  hold_data <- data.frame("query_id" = character(0), "subject_id" = character(0), "comparison" = character(0), "identity" = numeric(0))
  for (i in 1:(length(label_ids) - 1)){
    for (j in (i + 1):length(label_ids)){
      alignment <- pwalign::pid(pwalign::pairwiseAlignment(sequence_list[i], sequence_list[j], substitutionMatrix = "BLOSUM62", scoreOnly = FALSE))
      #print(alignment)
      hold_data <- rbind(hold_data, data.frame("query_id" = label_ids[i], "subject_id" = label_ids[j], "comparison" = comparison, "identity" = alignment))
    }
  }
  return(hold_data)
}

######################################################################
#  plot of all input data by peptide immunogenicity
######################################################################

# distribution of peptide outcomes (colored by receptor; all workbook tabs)
peptide_distrubution <- load_training_ML_data %>%
  dplyr::filter(Immunogenicity %in% c("Immunogenic", "Non-Immunogenic", "Weakly Immunogenic")) %>%
  dplyr::mutate(
    Receptor_plot = dplyr::case_when(
      as.character(Receptor) %in% c("INR", "INR-like") ~ "INR",
      as.character(Receptor) %in% c("LNR", "LNR-like") ~ "LNR",
      as.character(Receptor) == "phytocytokine_receptor" ~ "phytocytokine",
      TRUE ~ as.character(Receptor)
    )
  ) %>%
  # PEPR / SCORE are held out of this summary plot
  dplyr::filter(!Receptor_plot %in% c("PEPR", "SCORE")) %>%
  group_by(Receptor_plot, Immunogenicity) %>%
  summarize(n = n(), .groups = "drop") %>%
  dplyr::rename(Receptor = Receptor_plot)

peptide_receptor_order <- c(
  "CuRe1", "CORE", "EFR", "FLS2", "FLS3", "INR", "LNR", "RLP23",
  "PERU", "RLP42", "MIK2", "NUT", "phytocytokine", "unknown"
)
peptide_distrubution$Receptor <- factor(
  peptide_distrubution$Receptor,
  levels = peptide_receptor_order
)
# Top-to-bottom after coord_flip: Immunogenic, Weakly Immunogenic, Non-Immunogenic
peptide_distrubution$Immunogenicity <- factor(
  peptide_distrubution$Immunogenicity,
  levels = c("Non-Immunogenic", "Weakly Immunogenic", "Immunogenic")
)

immunogenicity_distrubution <- ggplot(
    data = peptide_distrubution,
    aes(x = Immunogenicity, y = n, fill = Receptor)
  ) +
  geom_bar(stat = "identity", linewidth = 0, width = 0.9) +
  coord_flip() +
  theme_clean_thin(base_size = 5.5) +
  theme(axis.text.x = element_text(color = "black", size = 5),
        axis.text.y = element_text(color = "black", size = 5),
        axis.title.x = element_text(size = 5.5),
        plot.margin = margin(1, 4, 1, 1, "pt")) +
  labs(x = "", y = "Count") +
  scale_fill_manual(values = receptor_colors, breaks = peptide_receptor_order) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
  geom_text(data = peptide_distrubution %>%
              group_by(Immunogenicity) %>%
              summarise(n = sum(n), .groups = "drop"),
            aes(label = n, y = n, x = Immunogenicity),
            hjust = -0.15,
            inherit.aes = FALSE,
            size = 1.8,
            family = "Arial")

ggsave(filename = "./04_Preprocessing_results/peptide_distrubution.pdf", plot = immunogenicity_distrubution, device = "pdf", dpi = 300, width = 2.1, height = 0.7)


######################################################################
#  parse and compare homologs of receptors
######################################################################

# Shared receptor–epitope display order so both plots list pairs together
# Parent receptors only for ordered plots (-like variants remap to INR/LNR)
receptor_order <- c("CuRe1","CORE","EFR","FLS2","FLS3","INR","LNR","RLP23","PERU","RLP42","MIK2","NUT","phytocytokine","PEPR","SCORE","unknown")
epitope_order <- names(epitope_colors)    # crip21, csp22, elf18, flg22, ..., phytocytokine

# Sequence/length comparison plots use the Model_Building tab only
# (all-tab data is used for the peptide immunogenicity distribution above)
all_tabs_data <- load_training_ML_data
load_training_ML_data <- all_tabs_data %>%
  dplyr::filter(Source_Sheet == "Model_Building")
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

# LNR (+ LNR-like)
receptor_LNR <- rbind(
  load_training_ML_data %>% filter(Receptor == "LNR"),
  load_training_ML_data %>% filter(Receptor == "LNR-like")
)
receptor_LNR <- receptor_LNR[!duplicated(receptor_LNR$Receptor.Sequence),]
LNR_comparison <- identity_calc(receptor_LNR$Locus.ID.Genbank, receptor_LNR$Receptor.Sequence, "LNR")
LNR_comparison <- subset(LNR_comparison, query_id != subject_id)

# phytocytokine receptor
receptor_phytocytokine <- load_training_ML_data %>% filter(Receptor == "phytocytokine_receptor")
receptor_phytocytokine <- receptor_phytocytokine[!duplicated(receptor_phytocytokine$Receptor.Sequence),]
phytocytokine_receptor_comparison <- identity_calc(
  receptor_phytocytokine$Locus.ID.Genbank,
  receptor_phytocytokine$Receptor.Sequence,
  "phytocytokine"
)
phytocytokine_receptor_comparison <- subset(phytocytokine_receptor_comparison, query_id != subject_id)

# unknown receptors
receptor_unknown <- load_training_ML_data %>% filter(Receptor == "unknown")
receptor_unknown <- receptor_unknown[!duplicated(receptor_unknown$Receptor.Sequence),]
unknown_receptor_comparison <- identity_calc(
  receptor_unknown$Locus.ID.Genbank,
  receptor_unknown$Receptor.Sequence,
  "unknown"
)
unknown_receptor_comparison <- subset(unknown_receptor_comparison, query_id != subject_id)

# CuRe1: single unique sequence — no pairwise identity; mark as a point at 100%
receptor_CuRe1 <- load_training_ML_data %>% filter(Receptor == "CuRe1")
receptor_CuRe1 <- receptor_CuRe1[!duplicated(receptor_CuRe1$Receptor.Sequence),]
CuRe1_singleton <- data.frame(
  query_id = receptor_CuRe1$Locus.ID.Genbank[1],
  subject_id = receptor_CuRe1$Locus.ID.Genbank[1],
  comparison = "CuRe1",
  identity = 100,
  stringsAsFactors = FALSE
)

combine_receptor_comparison <- rbind(
  FLS2_comparison, PERU_comparison, MIK2_comparison, CORE_comparison, EFR_comparison,
  FLS3_comparison, INR_comparison, LNR_comparison, RLP42_comparison, RLP23_comparison,
  NUT_comparison, phytocytokine_receptor_comparison, unknown_receptor_comparison
)
combine_receptor_comparison$comparison <- factor(combine_receptor_comparison$comparison, levels = receptor_order)

# Sequence counts for labels (include singleton receptors)
receptor_seq_n <- load_training_ML_data %>%
  dplyr::mutate(
    comparison = dplyr::case_when(
      as.character(Receptor) %in% c("INR", "INR-like") ~ "INR",
      as.character(Receptor) %in% c("LNR", "LNR-like") ~ "LNR",
      as.character(Receptor) == "phytocytokine_receptor" ~ "phytocytokine",
      TRUE ~ as.character(Receptor)
    )
  ) %>%
  dplyr::filter(comparison %in% receptor_order) %>%
  dplyr::group_by(comparison) %>%
  dplyr::summarise(number = dplyr::n_distinct(Receptor.Sequence), .groups = "drop")
receptor_stats <- receptor_seq_n
receptor_stats$comparison <- factor(receptor_stats$comparison, levels = receptor_order)

receptor_singles <- CuRe1_singleton
receptor_singles$comparison <- factor(receptor_singles$comparison, levels = receptor_order)

receptor_sequence_comparison_plot <- ggplot(combine_receptor_comparison, aes(x = comparison, y = identity, fill = comparison)) +
  stat_ydensity(aes(color = comparison), alpha = 0.85, scale = "width", linewidth = 0.35) +
  geom_boxplot(fill = "white", width = 0.25, outlier.shape = NA, linewidth = 0.35) +
  geom_point(
    data = receptor_singles,
    aes(x = comparison, y = identity, color = comparison),
    size = 1.6,
    shape = 16,
    inherit.aes = FALSE
  ) +
  theme_clean_thin() +
  xlab("Homolog Comparisons") +
  ylab("Percent Identity")+
  theme(axis.text.x = element_text(angle = 45, hjust = 1, color = "black", size = 7), 
        axis.text.y = element_text(color = "black", size = 7)) +
  scale_x_discrete(limits = receptor_order, drop = FALSE) +
  scale_y_continuous(limits = c(0, 120), breaks = c(20,40,60,80,100)) +
  coord_flip() +
  scale_fill_manual(values = receptor_colors, breaks = receptor_order) +
  scale_color_manual(values = receptor_colors, breaks = receptor_order) +
  geom_text(data = receptor_stats, aes(x = comparison, y = 110, label = number), size = 2.5, family = "Arial")

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

# phytocytokine (keep all unique pairs; identity_calc already enumerates i < j)
epitope_phytocytokine <- load_training_ML_data %>% filter(Ligand == "phytocytokine")
epitope_phytocytokine <- epitope_phytocytokine[!duplicated(epitope_phytocytokine$Ligand.Sequence),]
phytocytokine_comparison <- identity_calc(
  epitope_phytocytokine$Ligand.Sequence,
  epitope_phytocytokine$Ligand.Sequence,
  "phytocytokine"
)
phytocytokine_comparison <- subset(phytocytokine_comparison, query_id != subject_id)

combine_epitope_comparison <- rbind(In11_comparison, flg22_comparison, Pep25_comparison, SCOOP_comparison, csp22_comparison, 
                                    elf18_comparison, flgII28_comparison, peppg_comparison, crip21_comparison, nlp_comparison,
                                    screw_comparison, phytocytokine_comparison)

# Sequence counts; epitopes with only 1 unique sequence → plot as a dot at 100%
epitope_seq_n <- load_training_ML_data %>%
  filter(Ligand %in% epitope_order) %>%
  group_by(Ligand) %>%
  summarise(number = n_distinct(Ligand.Sequence), .groups = "drop") %>%
  dplyr::rename(comparison = Ligand)
epitope_stats <- epitope_seq_n
epitope_stats$comparison <- factor(epitope_stats$comparison, levels = epitope_order)

epitope_singles <- epitope_stats %>%
  filter(number == 1) %>%
  mutate(identity = 100, query_id = NA_character_, subject_id = NA_character_) %>%
  select(query_id, subject_id, comparison, identity)
epitope_singles$comparison <- factor(epitope_singles$comparison, levels = epitope_order)

combine_epitope_comparison <- combine_epitope_comparison %>%
  filter(!comparison %in% as.character(epitope_singles$comparison))
combine_epitope_comparison$comparison <- factor(combine_epitope_comparison$comparison, levels = epitope_order)

epitope_sequence_comparison_plot <- ggplot(combine_epitope_comparison, aes(x = comparison, y = identity, fill = comparison)) +
  stat_ydensity(aes(color = comparison), alpha = 0.85, scale = "width", linewidth = 0.35) +
  geom_boxplot(fill = "white", width = 0.25, outlier.shape = NA, linewidth = 0.35) +
  geom_point(
    data = epitope_singles,
    aes(x = comparison, y = identity, color = comparison),
    size = 1.6,
    shape = 16,
    inherit.aes = FALSE
  ) +
  theme_clean_thin() +
  xlab("Variant Comparisons") +
  ylab("Percent Identity")+
  theme(axis.text.x = element_text(angle = 45, hjust = 1, color = "black", size = 7), 
        axis.text.y = element_text(color = "black", size = 7)) +
  scale_x_discrete(limits = epitope_order, drop = FALSE) +
  scale_y_continuous(limits = c(0, 120), breaks = c(20,40,60,80,100)) +
  coord_flip() +
  scale_fill_manual(values = epitope_colors, breaks = epitope_order) +
  scale_color_manual(values = epitope_colors, breaks = epitope_order) +
  geom_text(data = epitope_stats, aes(x = comparison, y = 110, label = number), size = 2.5, family = "Arial")

ggsave(filename = "./04_Preprocessing_results/epitope_sequence_comparison_plot.pdf", 
plot = epitope_sequence_comparison_plot, device = "pdf", dpi = 300, width = 2.5, height = 2.7)


######################################################################
#  parse and compare epitope variant lengths
######################################################################

# Collapse psn / fun into "unknown"; keep phytocytokine as its own category
length_plot_data <- load_training_ML_data %>%
  dplyr::mutate(
    Ligand_plot = dplyr::case_when(
      as.character(Ligand) %in% c("psn", "fun", "unknown") ~ "unknown",
      TRUE ~ as.character(Ligand)
    )
  )
epitope_colors_length <- c(epitope_colors, "unknown" = "#9e9e9e")
length_order <- c(names(epitope_colors), "unknown")
length_plot_data$Ligand_plot <- factor(length_plot_data$Ligand_plot, levels = length_order)

# ridge plot — keep theme_ridges spacing + light grey grid; no black box border
epitope_length_comparison_plot <- ggplot(length_plot_data, aes(x = Ligand.Length, y = Ligand_plot)) +
  ggridges::geom_density_ridges(aes(fill = Ligand_plot), 
      rel_min_height = 0.01, alpha = 0.85, scale = 1.5, linewidth = 0.3, color = "black") +
  xlab("Length") +
  ylab("") +
  scale_fill_manual(values = epitope_colors_length, breaks = length_order) +
  scale_y_discrete(limits = length_order, expand = c(0, 0)) +
  scale_x_continuous(expand = c(0, 0)) +
  theme_ridges(center = TRUE, font_size = 8, font_family = "Helvetica Neue", grid = TRUE, line_size = 0.3) +
  theme(
    legend.position = "none",
    panel.border = element_blank(),
    axis.line = element_blank(),
    axis.ticks = element_blank(),
    panel.grid.major = element_line(colour = "grey85", linewidth = 0.3),
    panel.grid.minor = element_blank(),
    axis.text = element_text(family = "Helvetica Neue", colour = "black", size = 7),
    axis.title = element_text(family = "Helvetica Neue", colour = "black", size = 8),
    plot.background = element_rect(fill = "white", colour = NA),
    panel.background = element_rect(fill = "white", colour = NA)
  )

ggsave(filename = "./04_Preprocessing_results/epitope_length_comparison_plot.pdf", 
plot = epitope_length_comparison_plot, device = "pdf", dpi = 300, width = 2, height = 2)


######################################################################
#  pairwise identity distribution: epitope pairs vs receptor pairs
######################################################################

# Pool all pairwise identities across ligands/receptors into two groups
pairwise_identity_distribution <- rbind(
  data.frame(identity = combine_epitope_comparison$identity, pair_type = "epitope pairs"),
  data.frame(identity = combine_receptor_comparison$identity, pair_type = "receptor pairs")
)
pairwise_identity_distribution$pair_type <- factor(
  pairwise_identity_distribution$pair_type,
  levels = c("epitope pairs", "receptor pairs")
)

# Cache for fast replot tweaks (avoids recomputing all pairwise alignments)
write.csv(
  pairwise_identity_distribution,
  "./04_Preprocessing_results/pairwise_identity_distribution.csv",
  row.names = FALSE
)

pair_type_colors <- c("epitope pairs" = "#b35c46", "receptor pairs" = "#5a7fad")

pairwise_identity_distribution_plot <- ggplot(
    pairwise_identity_distribution,
    aes(x = identity, y = after_stat(density), fill = pair_type)
  ) +
  geom_histogram(position = "identity", alpha = 0.6, binwidth = 2.5, linewidth = 0) +
  theme_clean_thin() +
  xlab("pairwise identity (%)") +
  ylab("density") +
  theme(
    legend.position = c(0.02, 0.98),
    legend.justification = c(0, 1),
    legend.title = element_blank(),
    legend.text = element_text(size = 6, color = "black", margin = margin(l = 1)),
    legend.key.size = unit(6, "pt"),
    legend.key.height = unit(6, "pt"),
    legend.key.width = unit(6, "pt"),
    legend.spacing.y = unit(1, "pt"),
    legend.margin = margin(0, 0, 0, 0),
    legend.background = element_blank(),
    axis.text = element_text(color = "black", size = 7)
  ) +
  scale_fill_manual(values = pair_type_colors) +
  scale_x_continuous(limits = c(0, 100), breaks = c(0, 20, 40, 60, 80, 100)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.22))) +
  # Single vertical leader from label down toward the 99–100% bar (gap above bar)
  annotate("segment", x = 98.75, xend = 98.75, y = 0.049, yend = 0.043,
           colour = "black", linewidth = 0.3) +
  annotate("text", x = 100, y = 0.051, label = "point-mutant\nseries",
           colour = "black", hjust = 1, vjust = 0, size = 2.5, family = "Arial")

ggsave(filename = "./04_Preprocessing_results/pairwise_identity_distribution_plot.pdf",
plot = pairwise_identity_distribution_plot, device = "pdf", dpi = 300, width = 2, height = 1.75)


######################################################################
#  receptor phylogeny (MAFFT + FastTree + ggtree)
#  See 08_model_analysis/05_run_receptor_phylogeny.sh and
#      08_model_analysis/05_receptor_phylogeny.R
#  Outputs: 04_Preprocessing_results/receptor_phylogeny/
######################################################################

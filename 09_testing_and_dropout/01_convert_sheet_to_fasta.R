#-----------------------------------------------------------------------------------------------
# Krasileva Lab - Plant & Microbial Biology Department UC Berkeley
# Author: Danielle M. Stevens
# Last Updated: 07/06/2020
# Script Purpose: 
# Inputs: 
# Outputs: 
#-----------------------------------------------------------------------------------------------

library(readxl)
library(tidyverse)

######################################################################
#  function to turn dataframe (where one column is the name and one column is the sequence)
#   into a fasta file
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
# filter through blast results, filter by annotation, and put into distict fasta files
######################################################################

load_training_ML_data <- readxl::read_xlsx(path = "./02_in_data/All_LRR_PRR_ligand_data.xlsx", sheet = "Data_for_dropout_test")
load_training_ML_data <- data.frame(load_training_ML_data)[1:12]

receptor_full_length <- data.frame("Locus_Tag_Name" = character(0), "Sequence" = character(0))
for (k in 1:nrow(load_training_ML_data)){
  receptor_full_length <- rbind(receptor_full_length, data.frame(
        "Locus_Tag_Name" = paste(paste(">",  load_training_ML_data$Plant.species[k], sep=""), 
                                 load_training_ML_data$Locus.ID.Genbank[k], 
                                 load_training_ML_data$Receptor[k], 
                                 sep = "|"),
        "Sequence" = load_training_ML_data$Receptor.Sequence[k])
      )
}
  
receptor_full_length <- receptor_full_length %>% distinct(Sequence, .keep_all = TRUE)
writeFasta(receptor_full_length, "./09_testing_and_dropout/dropout_case/receptor_full_length_dropout_test.fasta")


######################################################################
# filter through blast results, filter by annotation, and put into distict fasta files
######################################################################

load_training_ML_data <- readxl::read_xlsx(path = "./02_in_data/All_LRR_PRR_ligand_data.xlsx", sheet = "Model_Validation")
load_training_ML_data <- data.frame(load_training_ML_data)[1:12]

receptor_full_length <- data.frame("Locus_Tag_Name" = character(0), "Sequence" = character(0))
for (k in 1:nrow(load_training_ML_data)){
  receptor_full_length <- rbind(receptor_full_length, data.frame(
        "Locus_Tag_Name" = paste(paste(">",  load_training_ML_data$Plant.species[k], sep=""), 
                                 load_training_ML_data$Locus.ID.Genbank[k], 
                                 load_training_ML_data$Receptor[k], 
                                 sep = "|"),
        "Sequence" = load_training_ML_data$Receptor.Sequence[k])
      )
}
  
receptor_full_length <- receptor_full_length %>% distinct(Sequence, .keep_all = TRUE)
writeFasta(receptor_full_length, "./09_testing_and_dropout/validation_data_set/receptor_full_length_model_validation.fasta")


######################################################################
# filter through blast results, filter by annotation, and put into distict fasta files
######################################################################

load_training_ML_data <- readxl::read_xlsx(path = "./02_in_data/All_LRR_PRR_ligand_data.xlsx", sheet = "Ngou_data_few_shot")
load_training_ML_data <- data.frame(load_training_ML_data)[1:12]

receptor_full_length <- data.frame("Locus_Tag_Name" = character(0), "Sequence" = character(0))
for (k in 1:nrow(load_training_ML_data)){
  receptor_full_length <- rbind(receptor_full_length, data.frame(
        "Locus_Tag_Name" = paste(paste(">",  load_training_ML_data$Plant.species[k], sep=""), 
                                 load_training_ML_data$Locus.ID.Genbank[k], 
                                 load_training_ML_data$Receptor[k], 
                                 sep = "|"),
        "Sequence" = load_training_ML_data$Receptor.Sequence[k])
      )
}
  
receptor_full_length <- receptor_full_length %>% distinct(Sequence, .keep_all = TRUE)
writeFasta(receptor_full_length, "./09_testing_and_dropout/Ngou_2025_SCORE_data/receptor_full_length_ngou_test.fasta")




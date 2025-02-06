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

load_training_ML_data <- readxl::read_xlsx(path = "./in_data/All_LRR_PRR_ligand_data.xlsx")
load_training_ML_data <- data.frame(load_training_ML_data)[1:12]

# distribution of peptide outcomes
peptide_distrubution <- load_training_ML_data %>% group_by(Ligand, Immunogenicity) %>% summarize(n=n())
webr::PieDonut(peptide_distrubution, aes(Immunogenicity,Ligand, count=n))


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

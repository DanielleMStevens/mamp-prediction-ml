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

load_training_ML_data <- readxl::read_xlsx(path = "./in_data/All_LRR_PRR_ligand_data.xlsx")
load_training_ML_data <- data.frame(load_training_ML_data)[1:12]

# distribution of peptide outcomes
peptide_distrubution <- load_training_ML_data %>% group_by(Ligand, Immunogenicity) %>% summarize(n=n())
webr::PieDonut(peptide_distrubution, aes(Immunogenicity,Ligand, count=n))


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


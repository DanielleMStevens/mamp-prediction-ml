# extract domain sequence information from train_immuno_stratify.csv

library(tidyverse)

# load train_immuno_stratify.csv
train_immuno_stratify <- read.csv("./05_datasets/train_immuno_stratify.csv")

# prep Header_Name and Receptor Sequence for fasta file conversion and remove duplicates
fasta_data <- data.frame(Header_Name = paste(">", train_immuno_stratify$Header_Name, sep = ""),
                         Receptor_Sequence = train_immuno_stratify$Receptor.Sequence) %>%
              distinct(Header_Name, .keep_all = TRUE)

# write fasta file
write.table(fasta_data, file = "./08_model_analysis/LRR_domain_sequences_train_immuno_stratify.fasta", 
            quote = FALSE, row.names = FALSE, col.names = FALSE, sep = "\n")


# prep Header_Name and Receptor Sequence for fasta file conversion and remove duplicates
# Filter for FLS2 sequences only
fasta_data <- data.frame(Header_Name = paste(">", train_immuno_stratify$Header_Name, sep = ""),
                         Receptor_Sequence = train_immuno_stratify$Receptor.Sequence) %>%
              filter(grepl("FLS2", Header_Name)) %>%
              distinct(Header_Name, .keep_all = TRUE)

# write fasta file
write.table(fasta_data, file = "./08_model_analysis/LRR_domain_sequences_FLS2.fasta", 
            quote = FALSE, row.names = FALSE, col.names = FALSE, sep = "\n")
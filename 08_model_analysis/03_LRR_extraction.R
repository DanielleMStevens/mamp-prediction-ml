# extract domain sequence information from train_immuno_stratify.csv

# load train_immuno_stratify.csv
train_immuno_stratify <- read.csv("./05_datasets/train_immuno_stratify.csv")

# prep Header_Name and Receptor Sequence for fasta file conversion
fasta_data <- data.frame(Header_Name = paste(">", train_immuno_stratify$Header_Name, sep = ""),
                         Receptor_Sequence = train_immuno_stratify$Receptor.Sequence)

# write fasta file
write.table(fasta_data, file = "./08_model_analysis/LRR_domain_sequences_train_immuno_stratify.fasta", 
            quote = FALSE, row.names = FALSE, col.names = FALSE, sep = "\n")
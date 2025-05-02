# Load necessary libraries
library(readr)
library(dplyr)
library(ggplot2)

# Read the data
correct_df <- read_tsv("results/correct_classification_report.tsv")
misclass_df <- read_tsv("results/misclassification_report.tsv")

# color code for genera of interest
epitope_colors <- c("#b35c46","#e2b048","#ebd56d","#b9d090","#37a170","#86c0ce","#7d9fc6", "#32527B", "#542a64", "#232232","#D5869D")
names(epitope_colors) <- c("crip21","csp22","elf18","flg22","flgII-28","In11","nlp", "pep-25", "pg", "scoop","screw")

receptor_colors <- c("#b35c46","#e2b048","#ebd56d","#b9d090","#37a170","#86c0ce","#7d9fc6", "#32527B", "#542a64", "#232232","#D5869D")
names(receptor_colors) <- c("CuRe1","CORE","EFR","FLS2","FLS3","INR","RLP23", "PERU", "RLP42", "MIK2","NUT")


# --- Data Preparation ---

# Count receptor occurrences in correct classifications
correct_counts <- as.data.frame(correct_df %>% count(Receptor, name = "Count") %>% mutate(Type = "Correct"))
correct_counts$Count[correct_counts$Receptor == "INR"] <- correct_counts$Count[correct_counts$Receptor == "INR"] + correct_counts$Count[correct_counts$Receptor == "INR-like"]
correct_counts <- correct_counts[correct_counts$Receptor != "INR-like",]
misclass_counts <- as.data.frame(misclass_df %>%count(Receptor, name = "Count") %>% mutate(Type = "Misclassified"))
combined_counts <- bind_rows(correct_counts, misclass_counts)

# Make counts negative for correct classifications for plotting direction
combined_counts <- combined_counts %>% mutate(Plot_Count = ifelse(Type == "Correct", -Count, Count))

# Ensure Receptor is a factor for consistent ordering on the y-axis
# Order receptors by total count (descending) for better visualization
receptor_order <- combined_counts %>% group_by(Receptor) %>% summarise(Total_Count = sum(Count)) %>% arrange(desc(Total_Count)) %>% pull(Receptor)
combined_counts <- combined_counts %>% mutate(Receptor = factor(Receptor, levels = rev(receptor_order))) # rev() for top-down display

# --- Plotting ---
receptor_distribution_plot <- ggplot(combined_counts, aes(x = Plot_Count, y = Receptor, fill = Receptor)) +
  geom_col() +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") + # Add line at zero
  #labels = function(x) abs(x), 
  scale_x_continuous(name = "Number of Combinations",
          limits = c(-120, 80), breaks = seq(-120, 80, 40), labels = c(120, 80, 40, 0, 40, 80)) +
  labs(y = "Receptor") +
  theme_classic() +
  scale_fill_manual(values = receptor_colors) +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5, color = "black", size = 8),
        axis.text.y = element_text(color = "black", size = 8),
        axis.title.x = element_text(color = "black", size = 8),
        axis.title.y = element_text(color = "black", size = 8),
        plot.title = element_text(hjust = 0.5),
        legend.position = "none",
        plot.margin = unit(c(1, 0.5, 0.5, 0.5), "cm")) + # Add padding (top, right, bottom, left)
   geom_text(aes(label = Count, x = Plot_Count + ifelse(Plot_Count < 0, -2, 2)),
     hjust = ifelse(combined_counts$Plot_Count < 0, 1, 0), size = 2.5, color = "black") +
   annotate("text", x = -max(abs(combined_counts$Plot_Count)) * 0.8, y = length(receptor_order) + 0.5, 
      label = "Correct", hjust = 0, vjust = -1, size = 2.5, color = "black") +
   annotate("text", x = max(abs(combined_counts$Plot_Count)) * 0.8, y = length(receptor_order) + 0.5, 
      label = "Misclassified", hjust = 0.7, vjust = -1, size = 2.5, color = "black") +
   coord_cartesian(clip = "off") # Allows annotations outside plot area


ggsave(filename = "./08_Validation_data_plots/receptor_distribution_plot.pdf", plot = receptor_distribution_plot, 
device = "pdf", dpi = 300, width = 2.3, height = 2.7)

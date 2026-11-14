# Load necessary libraries
library(readr)
library(dplyr)
library(ggplot2)

# Get command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Check if correct number of arguments provided
if (length(args) != 2) {
  stop("Please provide paths to both the correct classification and misclassification files.
Usage: Rscript script.R <correct_classification_file> <misclassification_file>")
}

# Read the data from command line arguments
correct_df <- read_tsv(args[1])
misclass_df <- read_tsv(args[2])

# color code for genera of interest
epitope_colors <- c("#b35c46","#e2b048","#ebd56d","#b9d090","#37a170","#86c0ce","#7d9fc6", "#32527B", "#542a64", "#232232","#D5869D")
names(epitope_colors) <- c("crip21","csp22","elf18","flg22","flgII-28","In11","nlp", "pep-25", "pg", "scoop","screw")

receptor_colors <- c("#b35c46","#e2b048","#ebd56d","#b9d090","#37a170","#86c0ce","#c5b3d0","#c5b3d0","#7d9fc6", "#32527B", "#542a64", "#232232","#D5869D", "#db7407","#7a1840","#b3adad", "black")
names(receptor_colors) <- c("CuRe1","CORE","EFR","FLS2","FLS3","INR","LNR","LNR-like","RLP23", "PERU", "RLP42", "MIK2","NUT", "PR","PEPR","unknown", "SCORE")


# --- Data Preparation ---

# Short display labels for long receptor names
relabel_receptor <- function(x) {
  x <- as.character(x)
  x[x %in% c("phytocytokine_receptor", "phytokinine")] <- "PR"
  x
}

# Count receptor occurrences in correct classifications
correct_counts <- as.data.frame(correct_df %>% count(Receptor, name = "Count") %>% mutate(Type = "Correct"))
# Merge -like variants into the parent receptor for plotting
for (pair in list(c("INR", "INR-like"), c("LNR", "LNR-like"))) {
  parent <- pair[[1]]
  child <- pair[[2]]
  if (parent %in% correct_counts$Receptor && child %in% correct_counts$Receptor) {
    correct_counts$Count[correct_counts$Receptor == parent] <-
      correct_counts$Count[correct_counts$Receptor == parent] +
      correct_counts$Count[correct_counts$Receptor == child]
  } else if (child %in% correct_counts$Receptor && !(parent %in% correct_counts$Receptor)) {
    correct_counts$Receptor[correct_counts$Receptor == child] <- parent
  }
  correct_counts <- correct_counts[correct_counts$Receptor != child, ]
}
misclass_counts <- as.data.frame(misclass_df %>% count(Receptor, name = "Count") %>% mutate(Type = "Misclassified"))
for (pair in list(c("INR", "INR-like"), c("LNR", "LNR-like"))) {
  parent <- pair[[1]]
  child <- pair[[2]]
  if (parent %in% misclass_counts$Receptor && child %in% misclass_counts$Receptor) {
    misclass_counts$Count[misclass_counts$Receptor == parent] <-
      misclass_counts$Count[misclass_counts$Receptor == parent] +
      misclass_counts$Count[misclass_counts$Receptor == child]
  } else if (child %in% misclass_counts$Receptor && !(parent %in% misclass_counts$Receptor)) {
    misclass_counts$Receptor[misclass_counts$Receptor == child] <- parent
  }
  misclass_counts <- misclass_counts[misclass_counts$Receptor != child, ]
}
combined_counts <- bind_rows(correct_counts, misclass_counts) %>%
  mutate(Receptor = relabel_receptor(Receptor)) %>%
  group_by(Receptor, Type) %>%
  summarise(Count = sum(Count), .groups = "drop")

# Make counts negative for correct classifications for plotting direction
combined_counts <- combined_counts %>% mutate(Plot_Count = ifelse(Type == "Correct", -Count, Count))

# Ensure Receptor is a factor for consistent ordering on the y-axis
# Order receptors by total count (descending) for better visualization
receptor_order <- combined_counts %>% group_by(Receptor) %>% summarise(Total_Count = sum(Count)) %>% arrange(desc(Total_Count)) %>% pull(Receptor)
combined_counts <- combined_counts %>% mutate(Receptor = factor(Receptor, levels = rev(receptor_order))) # rev() for top-down display

# --- Plotting ---
# Pad x past the longest bar so bars clear the axis frame (e.g. >200 when max is 212)
max_count <- max(combined_counts$Count)
label_nudge <- max(max_count * 0.04, 3)
x_pad <- max_count + 90
x_breaks <- pretty(c(-x_pad, x_pad), n = 5)

receptor_distribution_plot <- ggplot(
    combined_counts,
    aes(x = Plot_Count, y = Receptor, fill = Receptor)
  ) +
  geom_col(width = 0.8) +
  geom_vline(
    xintercept = 0,
    linetype = "dashed",
    color = "black",
    linewidth = 0.2
  ) +
  geom_text(
    aes(
      label = Count,
      x = Plot_Count + ifelse(Plot_Count < 0, -label_nudge, label_nudge)
    ),
    hjust = ifelse(combined_counts$Plot_Count < 0, 1, 0),
    size = 2.1,
    color = "black"
  ) +
  # Section labels in the top margin (not by extending the y-axis past receptors)
  annotate(
    "text",
    x = -max_count * 0.6,
    y = Inf,
    label = "Correct",
    hjust = 0.5,
    vjust = -0.6,
    size = 2.1,
    color = "black"
  ) +
  annotate(
    "text",
    x = max_count * 0.8,
    y = Inf,
    label = "Misclassified",
    hjust = 0.5,
    vjust = -0.6,
    size = 2.1,
    color = "black"
  ) +
  scale_x_continuous(
    name = "Number of combinations",
    limits = c(-x_pad, x_pad),
    breaks = x_breaks,
    labels = abs(x_breaks),
    expand = expansion(mult = 0)
  ) +
  scale_y_discrete(
    name = "Receptor",
    # Tight padding only — no empty axis stub past the last receptor
    expand = expansion(add = 0.35)
  ) +
  scale_fill_manual(values = receptor_colors, guide = "none") +
  theme_classic(base_line_size = 0.25) +
  theme(
    axis.line = element_line(color = "black", linewidth = 0.25),
    axis.ticks = element_line(color = "black", linewidth = 0.25),
    axis.text.x = element_text(angle = 0, hjust = 0.5, color = "black", size = 6),
    axis.text.y = element_text(color = "black", size = 6),
    axis.title.x = element_text(color = "black", size = 7),
    axis.title.y = element_text(color = "black", size = 7),
    legend.position = "none",
    plot.margin = unit(c(0.55, 0.4, 0.4, 0.4), "cm")  # room for Correct/Misclassified above
  ) +
  coord_cartesian(xlim = c(-x_pad, x_pad), clip = "off")

# Get directory of input file from command line args
input_dir <- dirname(args[1])
ggsave(
  filename = file.path(input_dir, "receptor_distribution_plot.pdf"),
  plot = receptor_distribution_plot,
  device = "pdf",
  dpi = 300,
  width = 1.9,
  height = 1.8
)

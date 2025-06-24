# Load the ggplot2 library
library(ggplot2)
library(readxl)
library(reshape2)
library(tidyverse)

# Read the Excel file from the same directory
data <- as.data.frame(readxl::read_xlsx(path = "./09_testing_and_dropout/ROS_screen_plots/Summary_csp22_validation_data.xlsx"))
save_sol_name <- data[1,]
data <- data[-1,]
data <- reshape2::melt(data, id.vars = c("Sol Number"))
colnames(data) <- c("Epitope", "Sol_Number", "value")
data$value <- as.numeric(data$value)

# 1. Handle negative values for size: Replace values < 0 with 1
# We create a new column for size to keep the original 'value' for coloring
data$size_mapped <- ifelse(data$value < 0, 1, data$value)

# Define the colors
data$color_category <- cut(data$value, breaks = c(-Inf, 14999, 75000, Inf), labels = c("0-14,999", "15,000-74,999", ">75,000"), right = TRUE)
color_mapping <- c("0-14,999" = "#8B0000", "15,000-74,999" = "#00008B", ">75,000" = "#4A4A4A")

# --- Create the Bubble Plot ---
bubble_plot <- ggplot(data, aes(x = Sol_Number, y = Epitope)) +
  geom_point(aes(size = size_mapped, fill = color_category), shape = 21, color = "black", alpha = 0.6) +
  scale_size_continuous(range = c(0.05, 8), guide = "none") +
  scale_fill_manual(values = color_mapping,name = "Avg. Max. RLUs") +
  ylab("Epitope") +
  xlab("Species") +
  theme_classic() +
  coord_flip() +
  theme(legend.position = "right", 
    axis.text.x = element_text(angle = 90, hjust = 1, size = 8, color = "black"),
    axis.text.y = element_text(size = 8, color = "black"),
    axis.title.x = element_text(size = 10, color = "black"),
    axis.title.y = element_text(size = 10, color = "black"),
    panel.border = element_rect(color = "black", fill = NA),
    panel.grid.minor = element_line(color = "grey90", size = 0.2),
    panel.grid.major = element_line(color = "grey90", size = 0.2))


# If you want to save the plot:
ggsave("./09_testing_and_dropout/ROS_screen_plots/ROS_bubble_plot.pdf", plot = bubble_plot, width = 10, height = 2.2, dpi = 300)


# Create summary plot of immunogenic outcomes grouped by Sol_Number
summary_plot <- ggplot(data[!is.na(data$value),], aes(x = Sol_Number, fill = factor(color_category, 
        levels = c(">75,000", "15,000-74,999", "0-14,999")))) +
  geom_bar(position = "dodge", color = "black", alpha = 0.6, linewidth = 0.3) +
  scale_fill_manual(values = color_mapping, name = "RLU Category") +
  ylim(0, 36) +
  theme_classic() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8, color = "black"),
    axis.text.y = element_text(size = 8, color = "black"), 
    axis.title.x = element_text(size = 9, color = "black"),
    axis.title.y = element_text(size = 9, color = "black"),
    legend.position = "none",
    panel.border = element_rect(color = "black", fill = NA),
    panel.grid.minor = element_line(color = "grey90", size = 0.2),
    panel.grid.major = element_line(color = "grey90", size = 0.2)
  ) +
  labs(x = "Species", y = "Count") +
  geom_text(stat = 'count', 
            aes(label = after_stat(count)), 
            position = position_dodge(width = 0.9),
            vjust = -0.5,
            size = 2.5)

# Save the summary plot
ggsave("./09_testing_and_dropout/ROS_screen_plots/ROS_summary_plot.pdf", plot = summary_plot, width = 1.9, height = 2.2, dpi = 300)


# ------------ also analyze Ngou et al. 2025 ROS screen data ------------

# Read excel file
# Read in data
data <- as.data.frame(readxl::read_xlsx(path = "./09_testing_and_dropout/Ngou_2025_SCORE_data/ROS_data/Ngou_ROS_raw_data.xlsx"))

# plot to rank ROS by immunogenic

# Create a data frame with row numbers for x-axis and split into immunogenic categories
data_ordered <- data %>% arrange(Average) %>% mutate(index = row_number())
remaining_data <- data_ordered$Average[data_ordered$Average > 20]
third_size <- length(remaining_data) / 2
thresholds <- sort(remaining_data)[c(ceiling(third_size))]

# Add immunogenicity category based on value ranges
data_ordered <- data_ordered %>% mutate(immunogenicity = case_when(Average <= 2 ~ "Non-immunogenic",
    Average <= thresholds[1] ~ "Weakly immunogenic", TRUE ~ "Immunogenic"))

# Create plot with colored points by category  
ggplot(data_ordered, aes(x = index, y = Average, color = immunogenicity)) +
  geom_point(size = 1.2, alpha = 0.5, stroke = 0) +
  geom_hline(yintercept = 2, linetype = "dashed", color = "gray50", alpha = 0.5) +
  geom_hline(yintercept = thresholds[1], linetype = "dashed", color = "gray50", alpha = 0.5) +
  scale_color_manual(values = c("Non-immunogenic" = "dark red",
                               "Weakly immunogenic" = "dark blue", 
                               "Immunogenic" = "grey")) +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8, color = "black"),
        axis.text.y = element_text(size = 8, color = "black"),
        axis.title = element_text(size = 9, color = "black"),
        panel.border = element_rect(color = "black", fill = NA),
        panel.grid = element_line(color = "grey90", size = 0.2),
        legend.position = "none",
        legend.title = element_blank(),
        legend.text = element_text(size = 8)) +
  labs(x = "Rank", y = "RLU Value")

# plot of homologs SCORE receptors

# Calculate mean RLU per receptor for ordering
receptor_means <- data %>% group_by(Receptor) %>% summarize(mean_rlu = mean(Average)) %>% arrange(mean_rlu)

# Convert Receptor to factor with ordered levels
data$Receptor <- factor(data$Receptor, levels = receptor_means$Receptor)

# Add immunogenicity classification to main data frame
data <- data %>% mutate(immunogenicity = case_when(Average <= 2 ~ "Non-immunogenic",
    Average <= thresholds[1] ~ "Weakly immunogenic", TRUE ~ "Immunogenic"))

ngou_boxplot <- ggplot(data, aes(x = Receptor, y = Average)) +
  stat_halfeye(aes(fill = Receptor), adjust = 0.5, width = 1, justification = -0.4, .width = 0, alpha = 0.5, stroke = NA) +
  geom_boxplot(aes(fill = Receptor), width = 0.25, outlier.shape = NA, alpha = 0.5, position = position_nudge(x = 0.2)) +
  geom_point(aes(color = immunogenicity), size = 1.2, alpha = 0.5, position = position_jitter(width = 0.1, seed = 1), stroke = 0) +
  scale_color_manual(values = c("Non-immunogenic" = "dark red",
                               "Weakly immunogenic" = "dark blue", 
                               "Immunogenic" = "grey")) +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8, color = "black"), axis.text.y = element_text(size = 8, color = "black"), axis.title = element_text(size = 9, color = "black"), panel.border = element_rect(color = "black", fill = NA), panel.grid = element_line(color = "grey90", size = 0.2)) +
  labs(x = "Receptor", y = "RLU Value") +
  guides(fill = "none", color = "none")

  
# Save boxplot
ggsave("./09_testing_and_dropout/ROS_screen_plots/Ngou_ROS_boxplot_SCORE_homologs.pdf", plot = ngou_boxplot, width = 4.5, height = 1.8, dpi = 300)

#chemcial feature plots

######################################################################
#  libraries to load
######################################################################

#load packages
library(readxl, warn.conflicts = FALSE, quietly = TRUE)
library(tidyverse, warn.conflicts = FALSE, quietly = TRUE)

# color code for genera of interest
epitope_colors <- c("#b35c46","#e2b048","#ebd56d","#b9d090","#37a170","#86c0ce","#7d9fc6", "#32527B", "#542a64", "#232232","#D5869D")
names(epitope_colors) <- c("crip21","csp22","elf18","flg22","flgII-28","In11","nlp", "pep-25", "pg", "scoop","screw")

receptor_colors <- c("#b35c46","#e2b048","#ebd56d","#b9d090","#37a170","#86c0ce","#7d9fc6", "#32527B", "#542a64", "#232232","#D5869D")
names(receptor_colors) <- c("CuRe1","CORE","EFR","FLS2","FLS3","INR","RLP23", "PERU", "RLP42", "MIK2","NUT")

######################################################################
#  load data
######################################################################

train_data <- readr::read_csv(file = "./05_datasets/train_data_with_all_train_immuno_stratify.csv")
Receptor_bulkiness_list <- data.frame("Receptor" = character(0), "Receptor Name" = character(0), "Postition" = numeric(0), 
        "Receptor_Bulkiness" = numeric(0), "Receptor_Charge" = numeric(0), "Receptor_Hydrophobicity" = numeric(0))
for (i in 1:nrow(train_data)){
    bulkiness_transformed <- as.vector(as.numeric(unlist(strsplit(train_data$Receptor_Bulkiness[i], ","))))
    charge_transformed <- as.vector(as.numeric(unlist(strsplit(train_data$Receptor_Charge[i], ","))))
    hydrophobicity_transformed <- as.vector(as.numeric(unlist(strsplit(train_data$Receptor_Hydrophobicity[i], ","))))
    position_list <- c("Position" = 1:length(bulkiness_transformed))
    Receptor_bulkiness_list <- rbind(Receptor_bulkiness_list, data.frame("Receptor" = rep(train_data$Receptor[i], length(bulkiness_transformed)), 
        "Receptor Name" = rep(train_data$`Receptor Name`[i], length(bulkiness_transformed)), 
        "Position" = position_list, 
        "Receptor_Bulkiness" = bulkiness_transformed, 
        "Receptor_Charge" = charge_transformed, 
        "Receptor_Hydrophobicity" = hydrophobicity_transformed))   
}


Epitope_feature_list <- data.frame("Epitope" = character(0), "Epitope_Sequence" = character(0), "Position" = numeric(0), 
        "Epitope_Bulkiness" = numeric(0), "Epitope_Charge" = numeric(0), "Epitope_Hydrophobicity" = numeric(0))

for (i in 1:nrow(train_data)){
    bulkiness_transformed <- as.vector(as.numeric(unlist(strsplit(train_data$Sequence_Bulkiness[i], ","))))
    charge_transformed <- as.vector(as.numeric(unlist(strsplit(train_data$Sequence_Charge[i], ","))))
    hydrophobicity_transformed <- as.vector(as.numeric(unlist(strsplit(train_data$Sequence_Hydrophobicity[i], ","))))
    position_list <- c("Position" = 1:length(bulkiness_transformed))
    Epitope_feature_list <- rbind(Epitope_feature_list, data.frame("Epitope" = rep(train_data$Epitope[i], length(bulkiness_transformed)), 
        "Epitope_Sequence" = rep(train_data$Sequence[i], length(bulkiness_transformed)), 
        "Position" = position_list, 
        "Epitope_Bulkiness" = bulkiness_transformed, 
        "Epitope_Charge" = charge_transformed, 
        "Epitope_Hydrophobicity" = hydrophobicity_transformed))   
}

######################################################################
#  plot PCA of chemical features - receptors
######################################################################

# Prepare data for PCA by calculating mean values per receptor
receptor_features <- Receptor_bulkiness_list %>%
  group_by(Receptor.Name) %>%
  summarize(
    Receptor_Bulkiness = mean(Receptor_Bulkiness),
    Receptor_Charge = mean(Receptor_Charge), 
    Receptor_Hydrophobicity = mean(Receptor_Hydrophobicity)
  ) %>%
  mutate(Receptor = Receptor_bulkiness_list$Receptor[match(Receptor.Name, Receptor_bulkiness_list$Receptor.Name)])

# Scale the features while keeping Receptor.Name
features_scaled <- receptor_features %>%
  select(-Receptor.Name, -Receptor) %>% # Remove non-numeric columns
  scale() %>%
  as.data.frame()

# Add Receptor.Name back to scaled data
features_scaled$Receptor.Name <- receptor_features$Receptor.Name
features_scaled$Receptor <- receptor_features$Receptor
# Perform PCA
pca_result <- prcomp(features_scaled[,1:3]) # Only use numeric columns

# Create dataframe for plotting with Receptor.Name preserved
pca_df <- as.data.frame(pca_result$x) %>%
  mutate(
    Receptor.Name = features_scaled$Receptor.Name,  # Add Receptor.Name back
    Receptor = features_scaled$Receptor             # Add Receptor back
  )

# Create PCA plot
# First check what data we have
print("Unique receptors in data:")
print(table(pca_df$Receptor))

pca_chemical_receptors <- ggplot(pca_df %>% filter(Receptor != "INR-like"), aes(x = PC1, y = PC2, color = Receptor)) +
  geom_point(size = 2, alpha = 0.6, stroke = 0) +
  scale_color_manual(values = receptor_colors) +
  theme_bw() +
  theme(legend.position = "none",
        axis.text.x = element_text(color = "black", size = 7),
        axis.text.y = element_text(color = "black", size = 7), 
        axis.title = element_text(color = "black", size = 8),
        panel.grid = element_blank()) +
  labs(x = paste0("PC1 (", round(summary(pca_result)$importance[2,1] * 100, 1), "%)"),
       y = paste0("PC2 (", round(summary(pca_result)$importance[2,2] * 100, 1), "%)"),
       title = "")

ggsave(filename = "./04_Preprocessing_results/Chemical_feature_analysis/pca_chemical_receptors.pdf", dpi = 300, plot = pca_chemical_receptors, width = 2, height = 1.5)


# --- same data but with umap instead of pca ----

# Prepare data for UMAP by calculating mean values per receptor
#receptor_features <- Receptor_bulkiness_list %>%
#  group_by(Receptor.Name) %>%
#  summarize(
#    Receptor_Bulkiness = mean(Receptor_Bulkiness),
#    Receptor_Charge = mean(Receptor_Charge), 
#    Receptor_Hydrophobicity = mean(Receptor_Hydrophobicity)
#  ) %>%
#  mutate(Receptor = Receptor_bulkiness_list$Receptor[match(Receptor.Name, Receptor_bulkiness_list$Receptor.Name)])

# Scale the features while keeping Receptor.Name
#features_scaled <- receptor_features %>%
#  select(-Receptor.Name, -Receptor) %>% # Remove non-numeric columns
#  scale() %>%
#  as.data.frame()

# Add Receptor.Name back to scaled data
#features_scaled$Receptor.Name <- receptor_features$Receptor.Name
#features_scaled$Receptor <- receptor_features$Receptor

# Perform UMAP
#umap_result <- umap::umap(features_scaled[,1:3]) # Only use numeric columns

# Create dataframe for plotting with Receptor.Name preserved
#umap_df <- as.data.frame(umap_result$layout) %>%
#  rename(UMAP1 = V1, UMAP2 = V2) %>%
#  mutate(
#    Receptor.Name = features_scaled$Receptor.Name,  # Add Receptor.Name back
#    Receptor = features_scaled$Receptor             # Add Receptor back
#  )

# Create UMAP plot
# First check what data we have
#print("Unique receptors in data:")
#print(table(umap_df$Receptor))

#umap_chemical_receptors <- ggplot(umap_df %>% filter(Receptor != "INR-like"), aes(x = UMAP1, y = UMAP2, color = Receptor)) +
#  geom_point(size = 2, alpha = 0.6, stroke = 0) +
#  scale_color_manual(values = receptor_colors) +
#  theme_bw() +
#  theme(legend.position = "none",
#        axis.text.x = element_text(color = "black", size = 7),
#        axis.text.y = element_text(color = "black", size = 7), 
#        axis.title = element_text(color = "black", size = 8),
#        panel.grid = element_blank()) +
#  labs(x = "UMAP1",
#       y = "UMAP2",
#       title = "")

#ggsave(filename = "./04_Preprocessing_results/Chemical_feature_analysis/umap_chemical_receptors.pdf", dpi = 300, plot = umap_chemical_receptors, width = 2, height = 1.5)




######################################################################
#  plot PCA of chemical features - epitopes 
######################################################################

# Prepare data for PCA by calculating mean values per epitope
epitope_features <- Epitope_feature_list %>%
  group_by(Epitope_Sequence) %>%
  summarize(
    Epitope_Bulkiness = mean(Epitope_Bulkiness),
    Epitope_Charge = mean(Epitope_Charge), 
    Epitope_Hydrophobicity = mean(Epitope_Hydrophobicity)
  ) %>%
  mutate(Epitope = Epitope_feature_list$Epitope[match(Epitope_Sequence, Epitope_feature_list$Epitope_Sequence)])

# Scale the features while keeping Epitope_Sequence
features_scaled <- epitope_features %>%
  select(-Epitope_Sequence, -Epitope) %>% # Remove non-numeric columns
  scale() %>%
  as.data.frame()

# Add Epitope_Sequence back to scaled data
features_scaled$Epitope_Sequence <- epitope_features$Epitope_Sequence
features_scaled$Epitope <- epitope_features$Epitope

# Perform PCA
pca_result <- prcomp(features_scaled[complete.cases(features_scaled[,1:3]),1:3]) # Only use numeric columns, removing NAs

# Create dataframe for plotting with Epitope_Sequence preserved
pca_df <- as.data.frame(pca_result$x) %>%
  mutate(
    Epitope_Sequence = features_scaled$Epitope_Sequence[complete.cases(features_scaled[,1:3])],  # Add Epitope_Sequence back, filtering NAs
    Epitope = features_scaled$Epitope[complete.cases(features_scaled[,1:3])]             # Add Epitope back, filtering NAs
  )

# Create PCA plot
# First check what data we have
print("Unique epitopes in data:")
print(table(pca_df$Epitope))

pca_chemical_epitopes <- ggplot(pca_df, aes(x = PC1, y = PC3, color = Epitope)) +
  geom_point(size = 2, alpha = 0.6, stroke = 0) +
  scale_color_manual(values = epitope_colors) +
  theme_bw() +
  theme(legend.position = "none",
        axis.text.x = element_text(color = "black", size = 7), 
        axis.text.y = element_text(color = "black", size = 7),
        axis.title = element_text(color = "black", size = 8),
        panel.grid = element_blank()) +
  labs(x = paste0("PC1 (", round(summary(pca_result)$importance[2,1] * 100, 1), "%)"),
       y = paste0("PC2 (", round(summary(pca_result)$importance[2,2] * 100, 1), "%)"),
       title = "")

ggsave(filename = "./04_Preprocessing_results/Chemical_feature_analysis/pca_chemical_epitopes.pdf", dpi = 300, plot = pca_chemical_epitopes, width = 2, height = 1.5)


# --- same data but with umap instead of pca ----

# Perform UMAP
library(uwot)
set.seed(42) # For reproducibility

# Prepare data for UMAP using the scaled features
umap_result <- uwot::umap(features_scaled[complete.cases(features_scaled[,1:3]),1:3], 
                    n_neighbors = 15,
                    min_dist = 0.1,
                    n_components = 2)

# Create dataframe for plotting
umap_df <- data.frame(
  UMAP1 = umap_result[,1],
  UMAP2 = umap_result[,2],
  Epitope_Sequence = features_scaled$Epitope_Sequence[complete.cases(features_scaled[,1:3])],
  Epitope = features_scaled$Epitope[complete.cases(features_scaled[,1:3])]
)

# Create UMAP plot
umap_chemical_epitopes <- ggplot(umap_df, aes(x = UMAP1, y = UMAP2, color = Epitope)) +
  geom_point(size = 2, alpha = 0.6, stroke = 0) +
  scale_color_manual(values = epitope_colors) +
  theme_bw() +
  theme(legend.position = "none",
        axis.text.x = element_text(color = "black", size = 7), 
        axis.text.y = element_text(color = "black", size = 7),
        axis.title = element_text(color = "black", size = 8),
        panel.grid = element_blank()) +
  labs(x = "UMAP1",
       y = "UMAP2",
       title = "")


ggsave(filename = "./04_Preprocessing_results/Chemical_feature_analysis/umap_chemical_epitopes.pdf", dpi = 300, plot = umap_chemical_epitopes, width = 2, height = 1.5)


######################################################################
#  tracking chemical features along the sequence - receptors
######################################################################

receptor_bulkiness_position_plot <- ggplot(Receptor_bulkiness_list, aes(x = Position, y = Receptor_Bulkiness, color = Receptor)) +
  geom_smooth(method = "gam", se = TRUE, formula = y ~ s(x, bs = "cs")) +
  facet_wrap(~Receptor, scales = "free_y") +
  theme_bw() +
  scale_color_manual(values = receptor_colors) +
  xlab("Sequence Length") + 
  ylab("Average Bulkiness") +
  theme(legend.position = "none",
  axis.text.x = element_text(color = "black", size = 7), 
  axis.text.y = element_text(color = "black", size = 7),
  panel.grid = element_blank())

ggsave(filename = "./04_Preprocessing_results/Chemical_feature_analysis/receptor_bulkiness_position_plot.pdf", dpi = 300, plot = receptor_bulkiness_position_plot, width = 4.5, height = 3)


receptor_charge_position_plot <- ggplot(Receptor_bulkiness_list, aes(x = Position, y = Receptor_Charge, color = Receptor)) +
  geom_smooth(method = "gam", se = TRUE, formula = y ~ s(x, bs = "cs")) +
  facet_wrap(~Receptor, scales = "free_y") +
  theme_bw() +
  scale_color_manual(values = receptor_colors) +
  xlab("Sequence Length") + 
  ylab("Average Charge") +
  theme(legend.position = "none",
  axis.text.x = element_text(color = "black", size = 7), 
  axis.text.y = element_text(color = "black", size = 7),
  panel.grid = element_blank())

ggsave(filename = "./04_Preprocessing_results/Chemical_feature_analysis/receptor_charge_position_plot.pdf", dpi = 300, plot = receptor_charge_position_plot, width = 4.5, height = 3)


receptor_hydrophobicity_position_plot <- ggplot(Receptor_bulkiness_list, aes(x = Position, y = Receptor_Hydrophobicity, color = Receptor)) +
  geom_smooth(method = "gam", se = TRUE, formula = y ~ s(x, bs = "cs")) +
  facet_wrap(~Receptor, scales = "free_y") +
  theme_bw() +
  scale_color_manual(values = receptor_colors) +
  xlab("Sequence Length") + 
  ylab("Average Hydrophobicity") +
  theme(legend.position = "none",
  axis.text.x = element_text(color = "black", size = 7), 
  axis.text.y = element_text(color = "black", size = 7),
  panel.grid = element_blank())

ggsave(filename = "./04_Preprocessing_results/Chemical_feature_analysis/receptor_hydrophobicity_position_plot.pdf", dpi = 300, plot = receptor_hydrophobicity_position_plot, width = 4.5, height = 3)

######################################################################
#  tracking chemical features along the sequence - receptors
######################################################################


epitope_bulkiness_position_plot <- ggplot(Epitope_feature_list, aes(x = Position, y = Epitope_Bulkiness, color = Epitope)) +
  geom_smooth(method = "gam", se = TRUE, formula = y ~ s(x, bs = "cs")) +
  facet_wrap(~Epitope, scales = "free") +
  theme_bw() +
  scale_color_manual(values = epitope_colors) +
  xlab("Sequence Length") + 
  ylab("Average Bulkiness") +
  theme(legend.position = "none",
  axis.text.x = element_text(color = "black", size = 7), 
  axis.text.y = element_text(color = "black", size = 7),
  panel.grid = element_blank())

ggsave(filename = "./04_Preprocessing_results/Chemical_feature_analysis/epitope_bulkiness_position_plot.pdf", dpi = 300, plot = epitope_bulkiness_position_plot, width = 4.5, height = 3)


epitope_charge_position_plot <- ggplot(Epitope_feature_list, aes(x = Position, y = Epitope_Charge, color = Epitope)) +
  geom_smooth(method = "gam", se = TRUE, formula = y ~ s(x, bs = "cs")) +
  facet_wrap(~Epitope, scales = "free") +
  theme_bw() +
  scale_color_manual(values = epitope_colors) +
  xlab("Sequence Length") + 
  ylab("Average Charge") +
  theme(legend.position = "none",
  axis.text.x = element_text(color = "black", size = 7), 
  axis.text.y = element_text(color = "black", size = 7),
  panel.grid = element_blank())

ggsave(filename = "./04_Preprocessing_results/Chemical_feature_analysis/epitope_charge_position_plot.pdf", dpi = 300, plot = epitope_charge_position_plot, width = 4.5, height = 3)


epitope_hydrophobicity_position_plot <- ggplot(Epitope_feature_list, aes(x = Position, y = Epitope_Hydrophobicity, color = Epitope)) +
  geom_smooth(method = "gam", se = TRUE, formula = y ~ s(x, bs = "cs")) +
  facet_wrap(~Epitope, scales = "free") +
  theme_bw() +
  scale_color_manual(values = epitope_colors) + 
  xlab("Sequence Length") + 
  ylab("Average Hydrophobicity") +
  theme(legend.position = "none",
  axis.text.x = element_text(color = "black", size = 7), 
  axis.text.y = element_text(color = "black", size = 7),
  panel.grid = element_blank())

ggsave(filename = "./04_Preprocessing_results/Chemical_feature_analysis/epitope_hydrophobicity_position_plot.pdf", dpi = 300, plot = epitope_hydrophobicity_position_plot, width = 4.5, height = 3)


######################################################################
#  tracking chemical features along the sequence - receptors
######################################################################

receptor_bulkiness_position_plot <- ggplot(Receptor_bulkiness_list, aes(x = Position, y = Receptor_Bulkiness, fill = Receptor)) +
  stat_summary(fun = mean, geom = "bar") +
  facet_wrap(~Receptor) +
  theme_bw() +
  ylim(0,22) +
  scale_fill_manual(values = receptor_colors) +
  xlab("Sequence Length") + 
  ylab("Average Bulkiness") +
  theme(legend.position = "none",
        axis.text.x = element_text(color = "black", size = 7), 
        axis.text.y = element_text(color = "black", size = 7),
        axis.title = element_text(color = "black", size = 8),
        strip.text = element_text(size = 6),
        panel.grid = element_blank())

ggsave(filename = "./04_Preprocessing_results/Chemical_feature_analysis/receptor_bulkiness_position_plot_v2.pdf", dpi = 300, plot = receptor_bulkiness_position_plot, width = 5, height = 2.8)


receptor_charge_position_plot <- ggplot(Receptor_bulkiness_list, aes(x = Position, y = Receptor_Charge, fill = Receptor)) +
  stat_summary(fun = mean, geom = "bar") +
  facet_wrap(~Receptor) +
  theme_bw() +
  ylim(-1.5,1.5) +
  geom_hline(yintercept = 0, size = 0.3, color = "grey80") +
  scale_fill_manual(values = receptor_colors) +
  xlab("Sequence Length") + 
  ylab("Average Charge") +
  theme(legend.position = "none",
        axis.text.x = element_text(color = "black", size = 7), 
        axis.text.y = element_text(color = "black", size = 7),
        axis.title = element_text(color = "black", size = 8),
        strip.text = element_text(size = 6),
        panel.grid = element_blank())

ggsave(filename = "./04_Preprocessing_results/Chemical_feature_analysis/receptor_charge_position_plot_v2.pdf", dpi = 300, plot = receptor_charge_position_plot, width = 5, height = 2.8)

receptor_hydrophobicity_position_plot <- ggplot(Receptor_bulkiness_list, aes(x = Position, y = Receptor_Hydrophobicity, fill = Receptor)) +
  stat_summary(fun = mean, geom = "bar") +
  facet_wrap(~Receptor) +
  theme_bw() +
  ylim(-0.2,2.8) +
  geom_hline(yintercept = 0, size = 0.3, color = "grey80") +
  scale_fill_manual(values = receptor_colors) +
  xlab("Sequence Length") + 
  ylab("Average Hydrophobicity") +
  theme(legend.position = "none",
        axis.text.x = element_text(color = "black", size = 7), 
        axis.text.y = element_text(color = "black", size = 7),
        axis.title = element_text(color = "black", size = 8),
        strip.text = element_text(size = 6),
        panel.grid = element_blank())

ggsave(filename = "./04_Preprocessing_results/Chemical_feature_analysis/receptor_hydrophobicity_position_plot_v2.pdf", dpi = 300, plot = receptor_hydrophobicity_position_plot, width = 5, height = 2.8)


######################################################################
#  tracking chemical features along the sequence - epitopes
######################################################################

epitope_bulkiness_position_plot <- ggplot(Epitope_feature_list, aes(x = Position, y = Epitope_Bulkiness, fill = Epitope)) +
  stat_summary(fun = mean, geom = "bar") +
  facet_wrap(~Epitope) +
  theme_bw() +
  ylim(0,22) +
  scale_fill_manual(values = epitope_colors) +
  xlab("Sequence Length") + 
  ylab("Average Bulkiness") +
  theme(legend.position = "none",
        axis.text.x = element_text(color = "black", size = 7), 
        axis.text.y = element_text(color = "black", size = 7),
        axis.title = element_text(color = "black", size = 8),
        strip.text = element_text(size = 6),
        panel.grid = element_blank())

ggsave(filename = "./04_Preprocessing_results/Chemical_feature_analysis/epitope_bulkiness_position_plot_v2.pdf", dpi = 300, plot = epitope_bulkiness_position_plot, width = 5, height = 2.8)


epitope_charge_position_plot <- ggplot(Epitope_feature_list, aes(x = Position, y = Epitope_Charge, fill = Epitope)) +
  stat_summary(fun = mean, geom = "bar") +
  facet_wrap(~Epitope) +
  theme_bw() +
  ylim(-1.5,1.5) +
  scale_fill_manual(values = epitope_colors) +
  xlab("Sequence Length") + 
  ylab("Average Charge") +
  geom_hline(yintercept = 0, size = 0.3, color = "grey80") +
  theme(legend.position = "none",
        axis.text.x = element_text(color = "black", size = 7), 
        axis.text.y = element_text(color = "black", size = 7),
        axis.title = element_text(color = "black", size = 8),
        strip.text = element_text(size = 6),
        panel.grid = element_blank())

ggsave(filename = "./04_Preprocessing_results/Chemical_feature_analysis/epitope_charge_position_plot_v2.pdf", dpi = 300, plot = epitope_charge_position_plot, width = 5, height = 2.8)

epitope_hydrophobicity_position_plot <- ggplot(Epitope_feature_list, aes(x = Position, y = Epitope_Hydrophobicity, fill = Epitope)) +
  stat_summary(fun = mean, geom = "bar") +
  facet_wrap(~Epitope) +
  theme_bw() +
  ylim(-0.2,2.8) +
  scale_fill_manual(values = epitope_colors) +
  xlab("Sequence Length") + 
  ylab("Average Hydrophobicity") +
  geom_hline(yintercept = 0, size = 0.3, color = "grey80") +
  theme(legend.position = "none",
        axis.text.x = element_text(color = "black", size = 7), 
        axis.text.y = element_text(color = "black", size = 7),
        axis.title = element_text(color = "black", size = 8),
        strip.text = element_text(size = 6),
        panel.grid = element_blank())

ggsave(filename = "./04_Preprocessing_results/Chemical_feature_analysis/epitope_hydrophobicity_position_plot_v2.pdf", dpi = 300, plot = epitope_hydrophobicity_position_plot, width = 5, height = 2.8)


######################################################################
#  tracking chemical features along the sequence - receptors
######################################################################


#ggplot(Receptor_bulkiness_list, aes(x = Receptor_Bulkiness, color = Receptor)) +
#geom_density() +
#scale_color_manual(values = receptor_colors) +
#theme_classic()

#ggplot(Receptor_bulkiness_list, aes(x = Receptor_Charge, color = Receptor)) +
#geom_density() +
#scale_color_manual(values = receptor_colors) +
#theme_classic()

#ggplot(Receptor_bulkiness_list, aes(x = Receptor_Hydrophobicity, color = Receptor)) +
#geom_density() +
#scale_color_manual(values = receptor_colors) +
#theme_classic()

#ggplot(Epitope_feature_list, aes(x = Epitope_Bulkiness, color = Epitope)) +
#geom_density() +
#scale_color_manual(values = epitope_colors) +
#theme_classic()

#ggplot(Epitope_feature_list, aes(x = Epitope_Charge, color = Epitope)) +
#geom_density() +
#scale_color_manual(values = epitope_colors) +
#theme_classic()

#ggplot(Epitope_feature_list, aes(x = Epitope_Hydrophobicity, color = Epitope)) +
#geom_density() +
#scale_color_manual(values = epitope_colors) +
#theme_classic()
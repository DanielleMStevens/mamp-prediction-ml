

bfactor_grape_rlks <- read.csv("11_grape_analysis/bfactor_winding_lrr_segments.csv")

bfactor_grape_rlks_sum <- bfactor_grape_rlks %>% group_by(Protein.Key) %>% summarise(max(LRR.Repeat.Number))
colnames(bfactor_grape_rlks_sum) <- c("Protein.Key", "LRR.Repeat.Number")
bfactor_grape_rlks_sum <-bfactor_grape_rlks_sum %>% arrange(LRR.Repeat.Number)

write.table(bfactor_grape_rlks_sum, "11_grape_analysis/bfactor_grape_rlks_sum.txt", row.names = FALSE, quote = FALSE, sep = "\t")

# count number of proteins with LRR.Repeat.Number > 15 & < 25
bfactor_grape_rlks_sum %>% filter(LRR.Repeat.Number >= 15 & LRR.Repeat.Number <= 25) %>% nrow()

# count number of proteins with LRR.Repeat.Number > 25
bfactor_grape_rlks_sum %>% filter(LRR.Repeat.Number > 25) %>% nrow()

# count number of proteins with LRR.Repeat.Number < 15 
bfactor_grape_rlks_sum %>% filter(LRR.Repeat.Number < 15) %>% nrow()

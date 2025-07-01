import pandas as pd
import os
from pathlib import Path

def setup_few_shot_directories():
    # Load original data
    data_df = pd.read_csv("09_testing_and_dropout/Ngou_2025_SCORE_data/data_Ngou_score_all.csv")
    
    sample_sizes = [2, 4, 8, 16, 32]
    
    for n_samples in sample_sizes:
        # Create directory
        dir_path = Path(f"05_datasets/few_shot_{n_samples}")
        dir_path.mkdir(exist_ok=True)
        
        # Create few-shot training data - sample exactly n_samples from the entire dataset
        few_shot_df = data_df.sample(n=n_samples, replace=False, random_state=42)
        
        # Save training data
        few_shot_df.to_csv(dir_path / f"train_data_{n_samples}_shot.csv", index=False)
        
        # Copy test data (same for all experiments)
        data_df.to_csv(dir_path / f"test_data_{n_samples}_shot.csv", index=False)
        
        print(f"Created {dir_path} with {len(few_shot_df)} training samples")
        print(f"  - Class distribution: {few_shot_df['Known Outcome'].value_counts().to_dict()}")

setup_few_shot_directories()
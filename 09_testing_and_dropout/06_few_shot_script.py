import pandas as pd
import os
from pathlib import Path

def setup_few_shot_directories():
    # Load original data
    data_df = pd.read_csv("09_testing_and_dropout/Ngou_2025_SCORE_data/data_Ngou_score_all.csv")
    
    # Define k-shot values and number of classes
    k_shots = [2, 4, 8, 16, 32]  # k samples per class
    n_classes = 3  # number of classes
    
    # Map string labels to numeric for easier handling
    label_mapping = {
        'Non-Immunogenic': 1,
        'Weakly Immunogenic': 2,
        'Immunogenic': 0
    }
    data_df['label'] = data_df['Known Outcome'].map(label_mapping)
    
    for k in k_shots:
        # Create directory
        dir_path = Path(f"05_datasets/few_shot_{k}")
        dir_path.mkdir(exist_ok=True)
        
        # Initialize empty list to store samples for each class
        few_shot_samples = []
        
        # Sample exactly k examples from each class
        for class_label in range(n_classes):
            class_df = data_df[data_df['label'] == class_label]
            
            # Check if we have enough samples for this class
            if len(class_df) < k:
                print(f"Warning: Not enough samples for class {class_label} for {k}-shot learning")
                print(f"Required: {k}, Available: {len(class_df)}")
                print("Using sampling with replacement to meet k-shot requirement")
                class_samples = class_df.sample(n=k, replace=True, random_state=42)
            else:
                class_samples = class_df.sample(n=k, replace=False, random_state=42)
            
            few_shot_samples.append(class_samples)
        
        # Combine samples from all classes
        few_shot_df = pd.concat(few_shot_samples, ignore_index=True)
        
        # Remove the numeric label column before saving
        few_shot_df = few_shot_df.drop('label', axis=1)
        
        # Save training data
        few_shot_df.to_csv(dir_path / f"train_data_{k}_shot.csv", index=False)
        
        # Save test data (excluding training samples)
        test_df = data_df[~data_df.index.isin(few_shot_df.index)].drop('label', axis=1)
        test_df.to_csv(dir_path / f"test_data_{k}_shot.csv", index=False)
        
        print(f"\nCreated {k}-shot learning dataset:")
        print(f"Training samples: {len(few_shot_df)} ({k} samples × {n_classes} classes)")
        print("Class distribution:")
        for outcome, count in few_shot_df['Known Outcome'].value_counts().items():
            print(f"  - {outcome}: {count}")

setup_few_shot_directories()
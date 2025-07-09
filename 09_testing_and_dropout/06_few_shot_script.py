import pandas as pd
import os
from pathlib import Path

def setup_few_shot_directories():
    # Load original data
    data_df = pd.read_csv("09_testing_and_dropout/Ngou_2025_SCORE_data/data_Ngou_score_all.csv")
    
    # Define k-shot values and number of classes
    k_shots = [2, 4, 8, 16, 32, 64, 128]  # k samples per class
    n_classes = 3  # number of classes
    
    # Map string labels to numeric for easier handling
    label_mapping = {
        'Non-Immunogenic': 1,
        'Weakly Immunogenic': 2,
        'Immunogenic': 0
    }
    data_df['label'] = data_df['Known Outcome'].map(label_mapping)
    
    # Create one consistent test set first
    test_samples = []
    
    # For each class, sample a balanced number for testing
    for class_label in range(n_classes):
        class_df = data_df[data_df['label'] == class_label]
        
        # Calculate test size for each class (1/3 of smallest class size)
        smallest_class_size = min([
            len(data_df[data_df['label'] == l]) 
            for l in range(n_classes)
        ])
        test_size_per_class = smallest_class_size // 3
        
        # Sample test data for this class
        test_class_samples = class_df.sample(
            n=test_size_per_class,
            replace=False,
            random_state=42
        )
        test_samples.append(test_class_samples)
    
    # Combine test samples from all classes
    test_df = pd.concat(test_samples, ignore_index=True)
    
    # Get indices of samples used in test set
    test_indices = test_df.index
    
    # Remove test samples from available training data
    train_data_df = data_df[~data_df.index.isin(test_indices)].copy()
    
    # Now create k-shot training sets
    for k in k_shots:
        # Create directory
        dir_path = Path(f"05_datasets/few_shot_{k}")
        dir_path.mkdir(exist_ok=True)
        
        # Initialize empty list to store samples for each class
        few_shot_samples = []
        
        # Sample exactly k examples from each class
        for class_label in range(n_classes):
            class_df = train_data_df[train_data_df['label'] == class_label]
            
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
        
        # Remove the label columns before saving
        few_shot_df = few_shot_df.drop('label', axis=1)
        test_df_save = test_df.drop('label', axis=1)
        
        # Save training data
        few_shot_df.to_csv(dir_path / f"train_data_{k}_shot.csv", index=False)
        
        # Save test data (same for all k)
        test_df_save.to_csv(dir_path / f"test_data_{k}_shot.csv", index=False)
        
        print(f"\nCreated {k}-shot learning dataset:")
        print(f"Training samples: {len(few_shot_df)} ({k} samples × {n_classes} classes)")
        print("Training class distribution:")
        for outcome, count in few_shot_df['Known Outcome'].value_counts().items():
            print(f"  - {outcome}: {count}")
    
    # Print test set information once (same for all k)
    print(f"\nTest set (same for all k-shots):")
    print(f"Test samples: {len(test_df)} (1/3 of smallest class size × {n_classes} classes)")
    print("Test class distribution:")
    for outcome, count in test_df['Known Outcome'].value_counts().items():
        print(f"  - {outcome}: {count}")

setup_few_shot_directories()
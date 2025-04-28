# datasets/seq_with_receptor_masked_dataset.py
import torch
import pandas as pd # Import pandas for type hinting if needed

# Mapping from category names to integer indices
category_to_index = {category: idx for idx, category in enumerate(["Immunogenic", "Non-Immunogenic", "Weakly Immunogenic"])}

class PeptideSeqWithReceptorMaskedDataset(torch.utils.data.Dataset):
    """
    Dataset class for peptide-receptor pairs, designed for use with
    receptor masking based on a 'Header_Name' column.

    It reads peptide sequence, receptor sequence, outcome label, and a
    'Header_Name' from a DataFrame. It derives a 'protein_key' from the
    'Header_Name' by replacing '|' with '_' for masking lookup purposes.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the dataset.

        Args:
            df (pd.DataFrame): DataFrame containing the data. Expected columns:
                               'Sequence' (peptide), 'Receptor Sequence',
                               'Known Outcome', 'Header_Name'.
        """
        # Verify required columns exist
        required_cols = ['Sequence', 'Receptor Sequence', 'Known Outcome', 'Header_Name']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in DataFrame: {missing_cols}")

        self.peptide_x = df['Sequence'].tolist() # Store as list for faster access?
        self.receptor_x = df['Receptor Sequence'].tolist()
        self.header_name = df['Header_Name'].tolist() # Store Header_Name
        # Apply mapping safely, handling potential missing categories if necessary
        self.y = df['Known Outcome'].map(category_to_index).fillna(-1).astype(int).tolist() # Map and handle unknowns
        if -1 in self.y:
            print("Warning: Some 'Known Outcome' values were not found in category_to_index and were mapped to -1.")

        self.name = "PeptideSeqWithReceptorMaskedDataset"
        print(f"Initialized {self.name} with {len(self.peptide_x)} samples.")


    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.peptide_x)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset by index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                  'peptide_x': The peptide sequence (str).
                  'receptor_x': The receptor sequence (str).
                  'y': The mapped outcome label (int).
                  'protein_key': Derived protein key (str) from Header_Name.
        """
        # Basic check for index validity
        if idx < 0 or idx >= len(self.peptide_x):
             raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self.peptide_x)}")

        header_name = self.header_name[idx]
        # Ensure header_name is a string before replacing
        protein_key = str(header_name).replace('|', '_') if pd.notna(header_name) else ""
        if not protein_key:
             print(f"Warning: Empty or NaN Header_Name found at index {idx}, resulting in empty protein_key.")


        return {
            'peptide_x': self.peptide_x[idx],
            'receptor_x': self.receptor_x[idx],
            'y': self.y[idx],
            'protein_key': protein_key # Add the derived protein key
        }

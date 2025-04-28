# models/esm_with_receptor_masked_model.py

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
# Import the base class - make sure the path is correct relative to this file
# If esm_with_receptor_model.py is in the same directory:
from .esm_with_receptor_model import ESMWithReceptorModel
# Or if it's one level up (e.g., in 'models/' directly):
# from esm_with_receptor_model import ESMWithReceptorModel

class ESMWithReceptorMaskedModel(ESMWithReceptorModel):
    """
    ESMWithReceptorModel with receptor sequence masking based on B-factor values.

    Inherits from ESMWithReceptorModel and overrides the collate_fn to
    mask specific residues in the receptor sequence before tokenization,
    based on negative B-factor values provided in a CSV file.

    Assumes the input batch provided to collate_fn contains a 'protein_key'
    derived from the dataset (e.g., from 'Header_Name' column).
    """
    def __init__(self, args=None, bfactor_csv_path="04_Preprocessing_results/bfactor_winding_lrr_segments.csv"):
        """
        Initializes the masked model.

        Args:
            args: Arguments passed from the training script (can be None).
            bfactor_csv_path (str): Path to the CSV file containing B-factor data.
                                     Expected columns: 'Protein Key', 'Residue Index',
                                     'Filtered B-Factor'.
        """
        super().__init__(args) # Initialize the base class

        # Load and preprocess B-factor data for masking
        self.bfactor_csv_path = bfactor_csv_path
        self.masking_info = self._load_masking_info()
        self.mask_token_id = self.tokenizer.mask_token_id
        if self.mask_token_id is None:
            raise ValueError("Tokenizer does not have a mask token. Cannot perform masking.")
        print(f"Using mask token ID: {self.mask_token_id}")


    def _load_masking_info(self):
        """Loads and processes the B-factor CSV to create a masking lookup."""
        masking_info = defaultdict(set)
        try:
            print(f"Loading B-factor masking info from: {self.bfactor_csv_path}")
            df = pd.read_csv(self.bfactor_csv_path)

            # Verify required columns in B-factor CSV
            required_bfactor_cols = ['Protein Key', 'Residue Index', 'Filtered B-Factor']
            missing_bfactor_cols = [col for col in required_bfactor_cols if col not in df.columns]
            if missing_bfactor_cols:
                 print(f"Warning: Missing required columns in B-factor CSV '{self.bfactor_csv_path}': {missing_bfactor_cols}. Masking might not work correctly.")
                 return masking_info # Return empty dict

            # Filter for negative B-factors
            df_filtered = df[df['Filtered B-Factor'] < 0].copy() # Use .copy() to avoid SettingWithCopyWarning

            # Ensure 'Residue Index' is numeric and handle potential errors
            df_filtered['Residue Index'] = pd.to_numeric(df_filtered['Residue Index'], errors='coerce')
            df_filtered.dropna(subset=['Residue Index'], inplace=True) # Drop rows where conversion failed
            df_filtered['Residue Index'] = df_filtered['Residue Index'].astype(int)

            # Group by Protein Key and collect 0-based Residue Indices to mask
            # Ensure 'Protein Key' is treated as string for consistent lookup
            for key, group in df_filtered.groupby('Protein Key'):
                masking_info[str(key)] = set(group['Residue Index'].tolist())

            print(f"Successfully loaded masking info for {len(masking_info)} proteins.")
            if not masking_info:
                 print("Warning: No negative B-factors found or no proteins could be processed. Masking will be disabled.")

        except FileNotFoundError:
            print(f"Error: B-factor CSV file not found at {self.bfactor_csv_path}. Masking disabled.")
        except Exception as e:
            print(f"Error loading or processing B-factor CSV '{self.bfactor_csv_path}': {e}. Masking disabled.")

        return masking_info

    def collate_fn(self, batch):
        """
        Custom collate function to tokenize and mask receptor sequences.

        Args:
            batch (list[dict]): A list of dictionaries, where each dictionary
                                 represents a sample and must contain:
                                 'peptide_x' (str): peptide sequence.
                                 'receptor_x' (str): receptor sequence.
                                 'protein_key' (str): key for receptor masking lookup.
                                 'y' (int/float): label.

        Returns:
            dict: A dictionary containing batched and tensorized inputs:
                  'x': {'peptide_x': {'input_ids': ..., 'attention_mask': ...},
                        'receptor_x': {'input_ids': ..., 'attention_mask': ...} (masked)}
                  'y': tensor of labels.
        """
        inputs = {}
        x_dict = {}

        # Basic validation of batch structure
        if not batch:
            return {'x': {}, 'y': torch.tensor([])} # Handle empty batch
        if not all(isinstance(item, dict) and 'peptide_x' in item and 'receptor_x' in item and 'protein_key' in item and 'y' in item for item in batch):
             # Find the problematic item for better debugging
             problematic_item = next((item for item in batch if not (isinstance(item, dict) and 'peptide_x' in item and 'receptor_x' in item and 'protein_key' in item and 'y' in item)), None)
             raise ValueError(f"Invalid batch item structure. Expected dict with keys 'peptide_x', 'receptor_x', 'protein_key', 'y'. Problematic item: {problematic_item}")


        # --- Peptide Processing (Standard) ---
        peptide_sequences = [example['peptide_x'] for example in batch]
        try:
            # Use truncation=True and max_length if sequences might be too long
            x_dict['peptide_x'] = self.tokenizer(
                peptide_sequences,
                return_tensors='pt',
                padding=True,
                truncation=True,
                # max_length=self.tokenizer.model_max_length # Optional: use tokenizer's max length
            )
        except Exception as e:
            print(f"Error tokenizing peptide sequences: {e}")
            # Handle error appropriately, maybe raise it or return empty batch
            raise

        # --- Receptor Processing (with Masking) ---
        protein_keys = [str(example['protein_key']) for example in batch] # Ensure keys are strings

        # Tokenize receptors individually to apply masks before padding
        try:
            tokenized_receptors = self.tokenizer(
                [example['receptor_x'] for example in batch],
                add_special_tokens=True, # Include CLS, SEP tokens
                return_tensors=None,     # Get lists of IDs first
                padding=False,           # Pad manually later
                truncation=True,         # Truncate if sequences exceed max length
                # max_length=self.tokenizer.model_max_length
            )
        except Exception as e:
            print(f"Error tokenizing receptor sequences: {e}")
            raise

        masked_input_ids_list = []
        num_masked_tokens = 0
        total_receptor_tokens = 0

        # Apply masking based on B-factors
        for i, receptor_input_ids in enumerate(tokenized_receptors['input_ids']):
            protein_key = protein_keys[i]
            indices_to_mask = self.masking_info.get(protein_key, set()) # Get indices for this key

            current_masked_ids = list(receptor_input_ids) # Mutable copy
            initial_token_count = len(current_masked_ids)
            total_receptor_tokens += (initial_token_count - 2) # Exclude CLS and SEP

            if indices_to_mask:
                num_residues_attempted_mask = 0
                num_residues_successfully_masked = 0
                # Add 1 to account for the [CLS] token at the beginning
                token_indices_to_mask = {idx + 1 for idx in indices_to_mask}

                for token_idx in token_indices_to_mask:
                    # Check bounds: ensure token_idx is within sequence length
                    # (excluding CLS at 0 and the last token, likely SEP)
                    if 0 < token_idx < (initial_token_count - 1):
                        num_residues_attempted_mask += 1
                        # Check if the token is not a special token before masking
                        original_token_id = current_masked_ids[token_idx]
                        if original_token_id not in self.tokenizer.all_special_ids:
                             current_masked_ids[token_idx] = self.mask_token_id
                             num_residues_successfully_masked += 1
                        # else: # Optional: print warning if trying to mask a special token
                        #     print(f"Warning: Skipped masking special token ID {original_token_id} at token index {token_idx} for protein {protein_key}")

                num_masked_tokens += num_residues_successfully_masked
                # Optional: Print masking stats per sequence if needed for debugging
                # if num_residues_attempted_mask > 0:
                #    print(f"Protein {protein_key}: Attempted mask={num_residues_attempted_mask}, Success={num_residues_successfully_masked}")


            masked_input_ids_list.append(current_masked_ids)

        if total_receptor_tokens > 0:
             print(f"Masked {num_masked_tokens} receptor tokens out of {total_receptor_tokens} total non-special tokens in this batch.")
        elif len(batch) > 0:
             print("No receptor tokens to mask in this batch (or sequences were empty).")


        # --- Manual Padding ---
        # Find the maximum length after potential truncation and masking
        max_len = max(len(ids) for ids in masked_input_ids_list) if masked_input_ids_list else 0

        padded_input_ids = []
        attention_masks = []

        for ids in masked_input_ids_list:
            padding_length = max_len - len(ids)
            # Pad sequence with pad token ID
            padded_ids = ids + [self.tokenizer.pad_token_id] * padding_length
            # Create attention mask (1 for real tokens, 0 for padding)
            attn_mask = [1] * len(ids) + [0] * padding_length

            padded_input_ids.append(padded_ids)
            attention_masks.append(attn_mask)

        # Store padded tensors in the batch dictionary
        if padded_input_ids:
            x_dict['receptor_x'] = {
                'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_masks, dtype=torch.long)
            }
        else: # Handle case where receptor sequences might be empty/problematic
             x_dict['receptor_x'] = {
                 'input_ids': torch.tensor([], dtype=torch.long),
                 'attention_mask': torch.tensor([], dtype=torch.long)
             }


        # --- Labels ---
        try:
             # Ensure labels are tensor-compatible (e.g., long for CE loss)
             inputs['y'] = torch.tensor([example['y'] for example in batch], dtype=torch.long)
        except Exception as e:
             print(f"Error converting labels to tensor: {e}")
             # Handle error appropriately, e.g., check label types
             raise

        inputs['x'] = x_dict

        return inputs

    # --- Inherited Methods ---
    # The following methods are inherited directly from ESMWithReceptorModel
    # and should work without modification unless the core logic involving
    # embeddings needs changes due to masking (which it typically shouldn't,
    # as the masking happens at the input token level).

    # forward(self, batch_x)
    # get_tokenizer(self)
    # batch_decode(self, batch)
    # get_pr(self, logits)
    # get_stats(self, gt, pr, train=False)

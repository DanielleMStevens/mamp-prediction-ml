from torchinfo import summary
import torch
import sys
from pathlib import Path

# Add the models directory to the Python path
script_dir = Path(__file__).parent
models_dir = script_dir / "models"
sys.path.append(str(models_dir))

# esm models - current models
from models.esm_with_receptor_model import ESMWithReceptorModel
from models.esm_all_chemical_features import ESMallChemicalFeatures
from models.esm_positon_weighted import BFactorWeightGenerator
from models.esm_positon_weighted import ESMBfactorWeightedFeatures

def get_model_summary():
    # Create a sample model
    model = ESMReceptorChemical(args=None, num_classes=3)
    model.eval()  # Set to evaluation mode
    
    # Create sample batch
    batch_size = 2
    seq_length = 256
    
    # Create input tensors
    sequence_ids = torch.zeros((batch_size, seq_length), dtype=torch.long)
    sequence_mask = torch.ones((batch_size, seq_length), dtype=torch.bool)
    receptor_ids = torch.zeros((batch_size, seq_length), dtype=torch.long)
    receptor_mask = torch.ones((batch_size, seq_length), dtype=torch.bool)
    seq_bulkiness = torch.zeros((batch_size,), dtype=torch.float)
    rec_bulkiness = torch.zeros((batch_size,), dtype=torch.float)
    
    # Create the input dictionary structure the model expects
    batch_x = {
        'peptide_x': {
            'input_ids': sequence_ids,
            'attention_mask': sequence_mask
        },
        'receptor_x': {
            'input_ids': receptor_ids,
            'attention_mask': receptor_mask
        },
        'sequence_bulkiness': seq_bulkiness,
        'receptor_bulkiness': rec_bulkiness
    }
    
    # Print summary
    summary(model, 
           input_data=[batch_x],  # Pass batch_x directly without wrapping in {'x': ...}
           col_names=["input_size", "output_size", "num_params", "trainable"],
           col_width=20,
           row_settings=["var_names"])

if __name__ == "__main__":
    get_model_summary()
import torch
import esm

class ESMModel:
    def __init__(self, model_name="esm2_t33_650M_UR50D"):
        """Initialize the ESM model.
        
        Args:
            model_name (str): Name of the ESM model to use.
                Default is "esm2_t33_650M_UR50D" which provides a good balance of speed and accuracy.
        """
        self.model_name = model_name
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.model.eval()  # Set to evaluation mode
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
    def encode_sequence(self, sequence, batch_tokens=False):
        """Encode a protein sequence using the ESM model.
        
        Args:
            sequence (str): Amino acid sequence to encode
            batch_tokens (bool): Whether to return batch tokens. Default is False.
            
        Returns:
            torch.Tensor: Sequence embeddings
        """
        # Prepare batch
        batch_converter = self.alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter([(0, sequence)])
        
        # Move to same device as model
        batch_tokens = batch_tokens.to(self.device)
        
        # Extract embeddings
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[self.model.num_layers])
            embeddings = results["representations"][self.model.num_layers]
        
        if batch_tokens:
            return embeddings, batch_tokens
        return embeddings
    
    def get_sequence_embeddings(self, sequence):
        """Get embeddings for a protein sequence.
        
        Args:
            sequence (str): Amino acid sequence
            
        Returns:
            torch.Tensor: Sequence embeddings (last layer)
        """
        embeddings = self.encode_sequence(sequence)
        return embeddings.cpu().numpy()  # Convert to numpy array on CPU 
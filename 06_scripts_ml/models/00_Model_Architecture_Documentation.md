# Model Architecture Documentation

## Current Models Overview

The three models represent a progression in complexity and capability:

1. **ESMWithReceptorModel**: Basic sequence-receptor interaction using CLS token embeddings
2. **ESMallChemicalFeatures**: Enhanced with chemical property integration
3. **ESMBfactorWeightedFeatures**: Most sophisticated, incorporating both chemical and structural information

This progression allows for increasingly complex feature integration while maintaining the core ESM2-based architecture.

## 1. ESMWithReceptorModel

### Base Architecture
- **Base Model**: ESM2-t30 (150M parameters) from Facebook AI
- **Input**: Peptide and receptor sequences
- **Architecture**:
  - Dual ESM encoders (shared weights, frozen) extracting CLS token embeddings
  - FiLM-based interaction module operating on CLS embeddings
  - Bidirectional modulation: `FiLM(peptide_CLS, receptor_CLS) + FiLM(receptor_CLS, peptide_CLS)`
  - 3-layer MLP classification head (`E -> E -> E//2 -> 3`)

### Key Components
- **FiLM Module**: Simple feature-wise linear modulation
  - Projects conditioning information to generate gamma and beta parameters
  - Applies affine transformation: `output = gamma * x + beta`
- **Classification Head**: 
  - Three linear layers with ReLU activations
  - Progressive dimension reduction: `E -> E -> E//2 -> 3`

### Features
- Symmetric peptide-receptor interaction using CLS token embeddings
- Frozen ESM2 backbone for stable feature extraction
- Comprehensive evaluation metrics (accuracy, F1, ROC AUC, PR AUC)

## 2. ESMallChemicalFeatures

### Base Architecture
- **Base Model**: ESM2-t30 with selective layer freezing
- **Input**: Combined peptide-receptor sequences with chemical features
- **Architecture**:
  - ESM2 encoder with last layer trainable (first 29 layers frozen)
  - Enhanced FiLMWithConcatenation module for chemical feature integration
  - Max pooling for sequence representation
  - 2-layer MLP with LayerNorm and Dropout

### Key Components
- **Chemical Feature Processing**:
  - Processes 3 chemical features per residue:
    - Bulkiness
    - Charge
    - Hydrophobicity
  - Projection network: `3 -> 64 -> feature_dim`
- **FiLMWithConcatenation**:
  - Layer normalization and dropout (0.1)
  - Chemical feature integration with sequence embeddings
  - Position-wise feature modulation

### Features
- Selective fine-tuning (only last layer trainable)
- Chemical feature integration at sequence level
- Label smoothing in loss function
- L2 regularization in training step
- Proper handling of variable-length sequences with masking

## 3. ESMBfactorWeightedFeatures

### Base Architecture
- **Base Model**: ESM2-t30 with selective fine-tuning
- **Input**: Sequences with B-factor and chemical features
- **Architecture**:
  - ESM2 encoder with configurable layer freezing
  - B-factor based position weighting
  - Enhanced FiLMWithConcatenation for feature integration
  - Comprehensive classification head

### Key Components
- **B-Factor Integration**:
  - BFactorWeightGenerator for position-specific weighting
  - Configurable weight range (default: 0.5 to 2.0)
  - Handles missing B-factor data gracefully
- **Chemical Feature Processing**:
  - Separate processing for sequence and receptor features
  - Enhanced projection network with LayerNorm
  - Position-wise feature modulation

### Features
- Structural information integration through B-factors
- Robust error handling for missing structural data
- Enhanced chemical feature processing
- Support for both training and evaluation modes
- Comprehensive evaluation metrics

## Common Features Across All Models

### Base Architecture
- All models use ESM2-t30 as the backbone
- Implement 3-class classification for immune response prediction
- Use FiLM-based feature modulation
- Support distributed training

### Training Features
- Comprehensive evaluation metrics
- Checkpoint saving and loading
- WandB integration for experiment tracking
- Proper handling of variable-length sequences

### Evaluation Metrics
- Accuracy
- Macro and weighted F1 scores
- ROC AUC (one-vs-rest)
- PR AUC for each class
- Prediction saving capabilities

### Data Processing
- Efficient batching with custom collate functions
- Support for chemical feature integration
- Proper sequence masking and padding
- Tokenization using ESM2 tokenizer


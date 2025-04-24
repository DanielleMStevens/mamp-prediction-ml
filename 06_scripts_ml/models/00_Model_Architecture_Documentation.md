# Model Architecture Documentation

## Base Models Overview

The codebase contains several model architectures that can be categorized into the following groups:

1.  Single Sequence Models
2.  Receptor-Peptide Interaction Models
3.  Specialized Models

## 1. Single Sequence Models

### AlphaFoldModel

-   **Input**: Pre-computed AlphaFold embeddings (256-dimensional)
-   **Architecture**:
    -   Adaptive average pooling (`nn.AdaptiveAvgPool2d((1, 1))`) for variable input sizes
    -   Two linear layers with ReLU activation (`Input -> Input -> 3`)
    -   Output: 3 classes
-   **Key Feature**: Processes pre-computed embeddings rather than raw sequences.
-   **Tokenizer**: `None` (uses pre-computed embeddings)
-   **Loss**: Cross-Entropy (implicitly handled by training engine)

### AMPModel

-   **Base**: AMPLIFY 350M parameter model (`chandar-lab/AMPLIFY_350M`)
-   **Input**: Single protein sequences
-   **Architecture**:
    -   Frozen AMPLIFY encoder
    -   First token embedding extraction (`hidden_states[-1][:, 0, :]`)
    -   3-layer MLP classifier (`E -> E -> E//2 -> 3`)
-   **Specialization**: Optimized for antimicrobial peptide analysis
-   **Tokenizer**: `AutoTokenizer.from_pretrained("chandar-lab/AMPLIFY_350M")`
-   **Loss**: Cross-Entropy (implicitly handled by training engine)

### ESMModel

-   **Base**: ESM-2 150M parameter model (`facebook/esm2_t30_150M_UR50D`)
-   **Input**: Single protein sequences
-   **Architecture**:
    -   Frozen ESM-2 encoder
    -   CLS token embedding (`last_hidden_state[:, 0, :]`)
    -   3-layer MLP classifier (`E -> E -> E//2 -> 3`)
-   **Key Feature**: Leverages evolutionary scale modeling
-   **Tokenizer**: `AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")`
-   **Loss**: `self.losses = ["ce"]` (Cross-Entropy)
-   **Evaluation**: `get_stats` calculates accuracy, F1, AUROC, and AUPRC.

### GLMModel

-   **Base**: gLM2 650M parameter model (`tattabio/gLM2_650M`)
-   **Input**: Protein sequences (prepended with special `'<+>'` token in `collate_fn`)
-   **Architecture**:
    -   Frozen gLM2 encoder
    -   First token embedding (`last_hidden_state[:, 0, :]`)
    -   3-layer MLP classifier (`E -> E -> E//2 -> 3`)
-   **Tokenizer**: `AutoTokenizer.from_pretrained("tattabio/gLM2_650M")`
-   **Loss**: Cross-Entropy (implicitly handled by training engine)

## 2. Receptor-Peptide Interaction Models

### ESMWithReceptorModel

-   **Base**: ESM-2 (`facebook/esm2_t30_150M_UR50D`)
-   **Input**: Peptide and receptor sequences
-   **Architecture**:
    -   Dual ESM encoders (shared weights, frozen) extracting CLS token embeddings (`[:, 0, :]`)
    -   FiLM-based interaction module (`FiLM` class) operating on CLS embeddings
    -   Bidirectional modulation: `FiLM(peptide_CLS, receptor_CLS) + FiLM(receptor_CLS, peptide_CLS)`
    -   3-layer MLP classification head (`E -> E -> E//2 -> 3`)
-   **Key Feature**: Symmetric peptide-receptor interaction using CLS token embeddings.
-   **Tokenizer**: `AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")`
-   **Loss**: `self.losses = ["ce"]` (Cross-Entropy)
-   **Evaluation**: `get_stats` calculates standard classification metrics.

### ESMWithReceptorAttnFilmModel

-   **Base**: ESM-2 (`facebook/esm2_t30_150M_UR50D`)
-   **Architecture**:
    -   Dual ESM encoders (shared weights, frozen) extracting full sequence embeddings (`last_hidden_state`)
    -   Attention-based FiLM module (`FiLMWithAttention` class)
    -   Full sequence attention between peptide (`query`) and receptor (`key/value`) embeddings.
    -   Modulates peptide embeddings based on attention context from receptor.
    -   Mean pooling of modulated peptide features (after FiLM application).
    -   3-layer MLP classification head (`E -> E -> E//2 -> 3`)
-   **Key Feature**: Sequence-level attention mechanism for interaction.
-   **Tokenizer**: `AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")`
-   **Loss**: `self.losses = ["ce"]` (Cross-Entropy)
-   **Evaluation**: `get_stats` calculates standard classification metrics.

### ESMReceptorChemical

-   **Base**: ESM-2 (`facebook/esm2_t30_150M_UR50D`)
-   **Architecture**:
    -   Partial model unfreezing: Embeddings and first 20 encoder layers frozen, last 10 layers trainable.
    -   `FiLMWithConcatenation` module:
        -   Processes full sequence embeddings.
        -   Integrates normalized bulkiness features (added to projected embeddings).
        -   Performs masked mean and max pooling on projected peptide and receptor features.
        -   Concatenates the four pooled features (`[peptide_max, peptide_mean, receptor_max, receptor_mean]`).
        -   Fuses concatenated features through a final MLP.
    -   Enhanced classifier MLP with LayerNorm, Dropout, and residual connections.
-   **Key Feature**: Incorporates chemical properties (bulkiness) and uses concatenation/pooling interaction.
-   **Tokenizer**: `AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")`
-   **Loss**: `nn.CrossEntropyLoss(label_smoothing=0.1)` (defined in `__init__`), plus L2 regularization added in `training_step`. `self.losses = ["ce"]`.
-   **Evaluation**: `get_stats` calculates standard classification metrics.

### GLMWithReceptorModel

-   **Base**: gLM2 (`tattabio/gLM2_650M`)
-   **Input**: Peptide and receptor sequences (prepended with `'<+>'` token in `collate_fn`)
-   **Architecture**:
    -   Dual gLM2 encoders (shared weights, frozen) extracting first token embeddings (`[:, 0, :]`)
    -   FiLM-based interaction (`FiLM` class): Modulates peptide embedding based on receptor embedding (`FiLM(peptide_token, receptor_token)`). *Note: Uni-directional modulation in contrast to ESMWithReceptorModel*.
    -   3-layer MLP classification head (`E -> E -> E//2 -> 3`)
-   **Key Feature**: Uses special '<+>' tokens for sequence marking and FiLM on first token embeddings.
-   **Tokenizer**: `AutoTokenizer.from_pretrained("tattabio/gLM2_650M")`
-   **Loss**: `self.losses = ["ce"]` (Cross-Entropy)
-   **Evaluation**: `get_stats` calculates standard classification metrics.

## 3. Specialized Models

### ESMContrastiveModel

-   **Base**: ESM-2 (`facebook/esm2_t30_150M_UR50D`)
-   **Architecture**:
    -   Frozen ESM-2 encoder extracting CLS token embedding (`[:, 0, :]`)
    -   Contrastive learning head (`self.net`) producing features for similarity loss.
    -   Classification head (`self.head`) operating on contrastive features.
    -   Supports both supervised (`ce`) and contrastive (`supcon`) losses (`self.losses = ['ce', 'supcon']`).
    -   `contrastive_output` flag controls whether forward returns only logits or `{"logits": ..., "features": ...}`.
-   **Key Feature**: Dual-objective learning (classification and similarity).
-   **Tokenizer**: `AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")`
-   **Evaluation**: `get_stats` calculates only accuracy (`acc@1`).

### ESMRegressionModel

-   **Base**: ESM-2 (`facebook/esm2_t30_150M_UR50D`)
-   **Architecture**:
    -   Frozen ESM-2 encoder extracting CLS token embedding (`[:, 0, :]`)
    -   Similar 3-layer MLP head as `ESMModel`, but with single-value output (`nn.Linear(E // 2, 1)`).
-   **Key Feature**: Predicts continuous values instead of classes.
-   **Tokenizer**: `AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")`
-   **Loss**: Not defined in model (likely MSE or similar, handled by training script).
-   **Evaluation**: Lacks a specific `get_stats` method.

### RandomForestBaselineModel

-   **Architecture**:
    -   Input features: Concatenated amino acid frequency vectors for peptide and receptor.
    -   `sklearn.ensemble.RandomForestClassifier` (`n_estimators=100`, `max_depth=10`).
-   **Key Feature**: Non-neural baseline model using simple composition features.
-   **Input Processing**: Converts sequences to amino acid frequency vectors using `_encode_sequence`.
-   **Training**: Uses a `fit` method.
-   **Evaluation**: `get_stats` calculates standard classification metrics.

## Single Sequence vs Combined Models

### Single Sequence Processing:
-   Models like `ESMModel`, `AMPModel`, `GLMModel` process one sequence at a time.
-   Use CLS token or first token embedding.
-   Simpler architecture but limited interaction modeling.

### Combined Sequence Processing:
-   Models like `ESMWithReceptorModel`, `GLMWithReceptorModel` process pairs.
-   Use various interaction mechanisms (FiLM, attention, concatenation/pooling).
-   Better suited for interaction prediction tasks.

## Feature Interaction Mechanisms

### FiLM (Feature-wise Linear Modulation):
-   **Basic FiLM (`FiLM` class)**: Used in `ESMWithReceptorModel` and `GLMWithReceptorModel`. Operates on CLS/first token embeddings. Generates `gamma`/`beta` from one embedding to modulate the other.
-   **FiLM with Attention (`FiLMWithAttention` class)**: Used in `ESMWithReceptorAttnFilmModel`. Operates on full sequence embeddings. Uses attention mechanism between sequences to derive context for modulation.
-   **FiLM with Concatenation (`FiLMWithConcatenation` class)**: Used in `ESMReceptorChemical`. Combines projection, pooling (mean/max), concatenation, and feature addition (bulkiness) before final fusion.

### Attention Mechanisms:
-   Used explicitly in `ESMWithReceptorAttnFilmModel` within the `FiLMWithAttention` module.
-   Allows for sequence-level interactions between peptide and receptor embeddings.
-   Supports variable length sequences through masking.

## Model-Specific Features

### Pre-trained Base Models:
-   ESM-2: Evolutionary scale modeling (`facebook/esm2_t30_150M_UR50D`, 150M parameters)
-   AMPLIFY: Specialized for antimicrobial peptides (`chandar-lab/AMPLIFY_350M`, 350M parameters)
-   gLM2: General protein language model (`tattabio/gLM2_650M`, 650M parameters)

### Special Tokens:
-   gLM2 models prepend `<+>` token during collation.
-   ESM models use standard protein tokens including `CLS`.
-   AMPLIFY uses specialized tokenization from its repository.

### Output Types:
-   Most models: 3-class classification logits.
-   `ESMRegressionModel`: Single continuous value.
-   `ESMContrastiveModel`: Classification logits + optional contrastive embeddings.
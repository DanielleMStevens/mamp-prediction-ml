# Prediction of MAMP-Receptor Interactions through a Protein Language Model

Table of contents:


## 00. Overview

This repository contains the code for the paper "A deep learning approach to epitope immunogenicity in plants" by Danielle Stevens, et al.

The code is broken down into the following: Prepping the data for training and validation, model hyperparameter optization and independent assessement. We additionally use the final weights to assess k-shot learning on a new dataset as well as use the model to find other receptors using three diffferent datasets (one for homolog variants, one for homolog variants with expanded recognition, and one for a convergently evolved receptor).

```
.
├── 01_LRR_Annotation/ # code base for LRR_Annotation
│   ├── analyze_bfactor_peaks.py
│   └── extract_lrr_sequences.py
├── 02_in_data/
│   └── All_LRR_PRR_ligand_data.xlsx
├── 03_out_data/
│   ├── lrr_annotation_plots/*.pdf # plots from LRR-Annotation on boundaries for LRR domain
│   ├── modeled_structures/
│   │   ├── pdb_for_lrr_annotator/*.pdb # converted pdb files from AlphaFold models for LRR-Annotation
│   │   ├── pdb_for_lrr_annotator/*_env/ # AlphaFold models output
│   │   └── alphafold_model_stats.txt # tracked output of alphafold stats
│   ├── training_data_summary/ # ignore for now
│   ├── lrr_annotation_results.txt # summary of LRR-Annotation domain extract from XX script
│   ├── lrr_domain_sequences.fasta # fasta file of just LRR domain from XX script
│   └── receptor_full_length.fasta # fasta file of full length receptor sequence from 06_scripts_ml/01_prep_receptor_sequences_for_modeling.R
├── 04_Preprocessing_results/
│   ├── Train_plots/
│   ├── Validation_data_plots/
│   ├── alphafold_scores.txt # summary of AlphaFold stats from alphafold_model_stats.txt
│   └── bfactor_winding_lrr_segments.txt # output of B-factor from LRR-Annotation
├── 05_datasets/*.csv # datasets used for model training and evaluation
├── 06_scripts_ml/
│   ├── models/ 
│   ├── losses/ # scripts for calculating loss
│   ├── datasets/ # scripts for managing datasets
│   ├── 01_prep_receptor_sequences_for_modeling.R # converts excel sheet to fasta file for receptor modeling
│   ├── 02_alphafold_to_lrr_annotation.py # summarizes AlphaFold and converts receptors for LRR-Annotation
│   ├── 03_parse_lrr_annotations.py # run and parse LRR-Annotation (extracts domain sequence)
│   ├── 04_data_prep_for_training.py # prep sequence data for model training (protein sequence only)
│   ├── 05_chemical_conversion.R # converts sequence data into chemical feature data in a format for model training/prediction
│   └── 06_main_train.py, engine_train.py, misc.py # main training scripts
├── 07_model_results/
│   ├── 00-12* models/ # models built and tested
│   ├── 00_visualize_model_predictions.R # plots the number of correct versus misclassfied interactions from validation data


```
## 01. Analyzing and visualizing input data

To illustrate the input data used for model training and testing, we assessed the diversity by both sequence and length. We used Biostrings (now pwalign) package to perform all-by-all global sequence similarity comparisons as well as visualized the length variation of protein ligands. 

```
# To make these plots, run the following command:
Rscript 08_model_analysis/00_visualize_input_data.R

# which will save all the plots generate in the following directory path:
├── 04_Preprocessing_results/
│   ├── peptide_distrubution.pdf
│   ├── receptor_sequence_comparison_plot.pdf
│   ├── epitope_sequence_comparison_plot.pdf
│   ├── epitope_length_comparison_plot.pdf


# To run the following script below will make similar plots for just training and validation
Rscript 08_model_analysis/01_visualize_train_data.R
Rscript 08_model_analysis/02_visualize_validation_data.R

# which will save all the plots in the following directories
├── 04_Preprocessing_results/
│   ├── Train_data_plots
│   ├── Validation_data_plots
```

We then wanted to visualize different metrics of our data processing pipeline (AlphaFold + LRR-Annotation). These commands are run below in [## 01. Prepping train + validation datasets], but the commands to visualize the outputs can be found here.

```
# To make the following plots, run the following command:
Rscript 08_model_analysis/03_visualize_data_pipeline.R 

# which will save all the plots generated in the following directory path:
├── 04_Preprocessing_results/
│   ├── pTM_plot.pdf
│   ├── pLDDT_pTM_scatter.pdf
│   ├── CORE_Bfactor.pdf
│   ├── INR_Bfactor.pdf
│   ├── max_winding_vs_lrrs.pdf
│   ├── lrr_repeat_vs_lrrs.pdf
```

Finally, after some interations we realized additional features would improve our model predictions. This includes different amino acid properties such as charge, hydrophobicity, and bulkiness. We wanted to visualize these properties and confirm that they provide information that can differentiate different receptors and protein ligands. Similar to Li et al., 2024, we made a PCA of the average values of the properities above for each receptor and ligand. 

```
# To make the following plots, run the following script:
Rscript 08_model_analysis/04_chemical_feature_plots.R 

# which will save all the plots generated in the following directory path:
├── 04_Preprocessing_results/
│   ├── Chemical_feature_analysis/
```


## 02. Prepping train + validation datasets

All the data for model training and validation is found in the excel sheet (All_LRR_PRR_ligand_data.xlsx) in 02_in_data. The file path is the following:
```
.
├── 02_in_data/
│   ├── All_LRR_PRR_ligand_data.xlsx
```

This data needs to be transformed in to a fasta file to run though AlphaFold in a semi-automated manner. 
```
# run this script on the command line 
Rscript 06_scripts_ml/01_prep_receptor_sequences_for_modeling.R

# it will generate a fasta file in the following file path:
.
├── 02_in_data/
│   ├── All_LRR_PRR_ligand_data.xlsx
```

Once the fasta file is generated, we will run it though local colabfold to generate predictive structures for each receptor sequence. To do so efficeintly, it is recommnded to run on a GPU (ideally NVIDIA). We initally used our local HPC, so the first few commands are for our HPC GPU's and may need to be modified for your system. This will generate AlphaFold structures (only one) for each unique receptor sequence.

```
# ------------------ each time a new gpu is started ------------------
module load anaconda3
conda activate localfold
module load gcc/10.5.0
export PATH="/global/scratch/users/dmstev/localcolabfold/colabfold-conda/bin:$PATH"

# ------------------ to run the model ------------------
colabfold_batch --num-models 1 ./03_out_data/receptor_full_length.fasta ./03_out_data/modeled_structures/receptor_only/

where: 
--num-models 1 means one model will be generate per sequence
./03_out_data/receptor_full_length.fasta is the input file 
./03_out_data/modeled_structures/receptor_only/ is the output directory path
```

Once all the structures are generated, we will prepare and run them through LRR-Annoation. 
```
# this script will generate some summary stats from AlphaFold models
python 06_scripts_ml/02_alphafold_to_lrr_annotation.py

# this will parse the lrr annotation outputs
python 06_scripts_ml/03_parse_lrr_annotations.py

# script will split the data depending on the 
python 06_scripts_ml/04_data_prep_for_training.py --split_type (immuno_stratify|random)
```

Once the data is split, we will then transform and add chemical feature data (amino acid bulkiness, charge, and hydrophobicity) before model training and evaluation.
```
# update names of train and test
Rscript 06_scripts_ml/05_chemical_conversion.R all train_input.csv test_input.csv

# for example, we can run the following for immuno_stratify ouptputs
Rscript 06_scripts_ml/05_chemical_conversion.R all train_immuno_stratify.csv test_immuno_stratify.csv
Rscript 06_scripts_ml/05_chemical_conversion.R all train_random.csv test_random.csv
```

Some models require tracking the b-factor, so we will run the command below.
```
# first run the script to analyze the bfactor peaks
python 01_LRR_Annotation/analyze_bfactor_peaks.py \
  --pdb-dir ./03_out_data/modeled_structures/pdb_for_lrr_annotator/ \
  -o ./04_Preprocessing_results/bfactor_winding_lrr_segments.csv

```
Finally let's assess sequence diversity by clustering and how this changes model performance.

```
# outputs all the epitopes into a fasta file
python -c "
import pandas as pd, pathlib
pathlib.Path('03_out_data').mkdir(exist_ok=True)
d = pd.read_excel('02_in_data/All_LRR_PRR_ligand_data.xlsx', sheet_name='Model_Building')
d['seq'] = d['Ligand Sequence'].astype(str).str.strip().str.upper()
d['fam'] = d['Ligand'].astype(str).str.strip().str.replace(r'[^A-Za-z0-9.-]+','_',regex=True)
u = d.drop_duplicates('seq').sort_values(['fam','seq'])
u['hdr'] = u['fam'] + '_' + (u.groupby('fam').cumcount()+1).astype(str).str.zfill(3)
assert u['hdr'].is_unique
with open('03_out_data/epitopes.fasta','w') as f:
    for h, s in zip(u['hdr'], u['seq']):
        f.write(f'>{h}\n{s}\n')
print(len(u), 'epitopes')
"

# then we will make a folder for the outputs of cd-hit and mmseqs
mkdir -p 03_out_data/cd_hit_outputs 03_out_data/mmseqs_outputs

# cd-hit runs
cd-hit -i 03_out_data/epitopes.fasta -o 03_out_data/cd_hit_outputs/epi_1.00 -c 1.00 -n 5 -G 1 -l 5 -g 1 -d 0 -M 0 -T 8
cd-hit -i 03_out_data/epitopes.fasta -o 03_out_data/cd_hit_outputs/epi_0.95 -c 0.95 -n 5 -G 1 -l 5 -g 1 -d 0 -M 0 -T 8
cd-hit -i 03_out_data/epitopes.fasta -o 03_out_data/cd_hit_outputs/epi_0.90 -c 0.90 -n 5 -G 1 -l 5 -g 1 -d 0 -M 0 -T 8
cd-hit -i 03_out_data/epitopes.fasta -o 03_out_data/cd_hit_outputs/epi_0.85 -c 0.85 -n 5 -G 1 -l 5 -g 1 -d 0 -M 0 -T 8
cd-hit -i 03_out_data/epitopes.fasta -o 03_out_data/cd_hit_outputs/epi_0.80 -c 0.80 -n 5 -G 1 -l 5 -g 1 -d 0 -M 0 -T 8
cd-hit -i 03_out_data/epitopes.fasta -o 03_out_data/cd_hit_outputs/epi_0.70 -c 0.70 -n 4 -G 1 -l 5 -g 1 -d 0 -M 0 -T 8
cd-hit -i 03_out_data/epitopes.fasta -o 03_out_data/cd_hit_outputs/epi_0.60 -c 0.60 -n 3 -G 1 -l 5 -g 1 -d 0 -M 0 -T 8
cd-hit -i 03_out_data/epitopes.fasta -o 03_out_data/cd_hit_outputs/epi_0.50 -c 0.50 -n 2 -G 1 -l 5 -g 1 -d 0 -M 0 -T 8

# mmseqs outputs
rm -rf 03_out_data/tmp
mmseqs easy-cluster 03_out_data/epitopes.fasta 03_out_data/mmseqs_outputs/mm_epi_1.00 03_out_data/tmp --min-seq-id 1.00 -c 0.8 --cov-mode 0 --seq-id-mode 0 --alignment-mode 3 --cluster-mode 0 -k 5 -s 7.5 --comp-bias-corr 0 --threads 8
rm -rf 03_out_data/tmp
mmseqs easy-cluster 03_out_data/epitopes.fasta 03_out_data/mmseqs_outputs/mm_epi_0.95 03_out_data/tmp --min-seq-id 0.95 -c 0.8 --cov-mode 0 --seq-id-mode 0 --alignment-mode 3 --cluster-mode 0 -k 5 -s 7.5 --comp-bias-corr 0 --threads 8
rm -rf 03_out_data/tmp
mmseqs easy-cluster 03_out_data/epitopes.fasta 03_out_data/mmseqs_outputs/mm_epi_0.90 03_out_data/tmp --min-seq-id 0.90 -c 0.8 --cov-mode 0 --seq-id-mode 0 --alignment-mode 3 --cluster-mode 0 -k 5 -s 7.5 --comp-bias-corr 0 --threads 8
rm -rf 03_out_data/tmp
mmseqs easy-cluster 03_out_data/epitopes.fasta 03_out_data/mmseqs_outputs/mm_epi_0.85 03_out_data/tmp --min-seq-id 0.85 -c 0.8 --cov-mode 0 --seq-id-mode 0 --alignment-mode 3 --cluster-mode 0 -k 5 -s 7.5 --comp-bias-corr 0 --threads 8
rm -rf 03_out_data/tmp
mmseqs easy-cluster 03_out_data/epitopes.fasta 03_out_data/mmseqs_outputs/mm_epi_0.80 03_out_data/tmp --min-seq-id 0.80 -c 0.8 --cov-mode 0 --seq-id-mode 0 --alignment-mode 3 --cluster-mode 0 -k 5 -s 7.5 --comp-bias-corr 0 --threads 8
rm -rf 03_out_data/tmp
mmseqs easy-cluster 03_out_data/epitopes.fasta 03_out_data/mmseqs_outputs/mm_epi_0.70 03_out_data/tmp --min-seq-id 0.70 -c 0.8 --cov-mode 0 --seq-id-mode 0 --alignment-mode 3 --cluster-mode 0 -k 5 -s 7.5 --comp-bias-corr 0 --threads 8
rm -rf 03_out_data/tmp
mmseqs easy-cluster 03_out_data/epitopes.fasta 03_out_data/mmseqs_outputs/mm_epi_0.60 03_out_data/tmp --min-seq-id 0.60 -c 0.8 --cov-mode 0 --seq-id-mode 0 --alignment-mode 3 --cluster-mode 0 -k 5 -s 7.5 --comp-bias-corr 0 --threads 8
rm -rf 03_out_data/tmp
mmseqs easy-cluster 03_out_data/epitopes.fasta 03_out_data/mmseqs_outputs/mm_epi_0.50 03_out_data/tmp --min-seq-id 0.50 -c 0.8 --cov-mode 0 --seq-id-mode 0 --alignment-mode 3 --cluster-mode 0 -k 5 -s 7.5 --comp-bias-corr 0 --threads 8
rm -rf 03_out_data/tmp
mmseqs easy-cluster 03_out_data/epitopes.fasta 03_out_data/mmseqs_outputs/mm_epi_0.40 03_out_data/tmp --min-seq-id 0.40 -c 0.8 --cov-mode 0 --seq-id-mode 0 --alignment-mode 3 --cluster-mode 0 -k 5 -s 7.5 --comp-bias-corr 0 --threads 8
rm -rf 03_out_data/tmp

# plot the clustering outputs
Rscript 08_model_analysis/08_clustering_plots.R

```
We will also want to plot sequence variation versus structural variation:
```
# install via conda tm-align
conda install tmalign

# run the script that will run tm-align as well as plot the results in comparison to their sequence similarlty
TMALIGN_WORKERS=12 Rscript 08_model_analysis/09_sequence_vs_structure.R

```



## 03. Model training and assessment

We will then need to edit the main training script for which file to train and test with and run each model as described in model_train_commands.sh

```
.
├── 06_scripts_ml/
│   ├── model_train_commands.sh
│   ├── 06_main_train.py
```
For each model, we ran two additional script to evaluate their performance. One assess the number of correct and misclassified perdictions based on the true label. The second make a confusion matrix of predicted and true labels of each class (immunogenic, weakly immunogenic, and non-immunogenic).

```
# below is an example however each one is detailed in model_train_command.sh
# the model name will need to be swapped as well as the input data for the associated 
# model training/test schema

python 07_model_results/01_make_confusion_matrix.py \
    --predictions_path 07_model_results/10_esm2_t33_650M_UR50D_last_layer_only_esm2_bfactor_weighted/test_preds.pth \
    --output_dir 07_model_results/10_esm2_t33_650M_UR50D_last_layer_only_esm2_bfactor_weighted \
    --data_info_path 05_datasets/test_data_with_all_test_immuno_stratify.csv

Rscript 07_model_results/00_visualize_model_predictions.R \
    07_model_results/04_immuno_stratify_esm2_all_chemical_features/correct_classification_report.tsv \
    07_model_results/04_immuno_stratify_esm2_all_chemical_features/misclassification_report.tsv
```

## 04. Final model assessment 

Once we determined which model architecture and hyperparameters performed best, we want to evalute it through two cases, an independent functional dataset as well as one receptor dropout case. We built this model with the intention of using it as an in silico screen for receptor-ligand variants but not new receptor-ligand pairs. To assess its performance for this task, we collected independent immunogenicity data by screening new CORE receptor variants with our library of csp22 variants via ROS production.

We can first visualize that data by running the script below:
```
Rscript 09_testing_and_dropout/00_ROS_test_data.R 

# which will save all the plots generated in the following directory path:
├── 09_testing_and_dropout/
│   ├── ROS_screen_plots/
```

Using that data, we can independently assess how our model is performing. To do so, we made additional scripts which will process this data for model prediction.

```
# Run the following scripts on the command line:
Rscript 09_testing_and_dropout/01_convert_sheet_to_fasta.R 
```
This will prep the receptor sequences into a fasta file. We will then need to run alphafold to model each receptor sequence and then run a version of LRR-Annotation to parse out the ectodomain. Activate the conda environment for ESM-2 and run the following command on a computer with a Nivida GPU. If a GPU is unavailable, local co-labfold can be run on a CPU but will need some minor modifications to the script. Refer the to localcolabfold GitHub for reference.

```
# Run the following command:
colabfold_batch --num-models 1 ./09_testing_and_dropout/test_data_set/receptor_full_length_model_validation.fasta ./09_testing_and_dropout/test_data_set/receptor_only/
```
After generating the structural models, we will process them to be compatible with colabfold parsing steps as well as generate the bfactor winding values for the model input. Run this on the main root director.

```
# This script will process the predicted PDB structures and output the required bfactor winding values as a CSV file, which can then be used in later data processing and model inference steps.

python 01_LRR_Annotation/analyze_bfactor_peaks.py \
    --pdb-dir ./09_testing_and_dropout/test_data_set/pdb_for_lrr_annotator/ \
    -o ./09_testing_and_dropout/test_data_set/bfactor_winding_lrr_segments.csv
```

Then run the commands from main_train_commands.sh and run the model.

```
Rscript 07_model_results/04_embeddings_analysis.R \
  --seqonly 09_testing_and_dropout/validation_data_set/model_02_test_predictions.csv \
  --final 09_testing_and_dropout/validation_data_set/model_00_test_predictions.csv \
  --output_dir 07_model_results/embeddings_comparison
```

## 05. Zero vs. few shot on SCORE-csp22

Now that we have a working model and framework, we wanted to assess how accurate our model was to a new receptor-ligand pair, SCORE-csp22. For context, SCORE is a convergently evolved LRR-RLK that also recognizes the csp22 ligand yet its sequence is highly diverge from CORE. This provides an excellent test case to assess how model's ability to generalize to new receptor-ligand pairs.

First, all SCORE data was split into three main groups: orthologs, LRR swaps, and AA substitutions. We then ran these through our mamp-ml model (zero shot) using the live co-lab notebook online to get predictions and were copied over to an excel sheet (Ngou_zero_shot_case.xlsx).

```
# first we will convert our excel sheet data into a fasta file to run through AlphaFold and LRR-Annotation
Rscript 09_testing_and_dropout/01_convert_sheet_to_fasta.R 
```

Then we will run AlphaFold on the fasta file. Since we have access to an HPC, we will spin up a GPU and run the following commands:
```
# This will activate 
module load anaconda3
conda activate localfold
module load gcc/10.5.0
export PATH="/global/scratch/users/dmstev/localcolabfold/colabfold-conda/bin:$PATH"

colabfold_batch --num-models 1 ./09_testing_and_dropout/Ngou_2025_SCORE_data/receptor_full_length_ngou_test.fasta ./09_testing_and_dropout/Ngou_2025_SCORE_data/receptor_only/
```

This will generate structural models of each receptor via AlphaFold2 and store them plus their scores and metrics in the receptor_only folder. Once we have these structures, we will run the below python scripts to clean up the data and run the top structures through LRR-Annotation to determine the ectodomain receptor sequence.

```
## update line 149, 178, 186, 189, 190 first before running.
python 09_testing_and_dropout/02_alphafold_to_lrr_annotation_test.py ./09_testing_and_dropout/Ngou_2025_SCORE_data/receptor_only/log.txt

## then update lines 114, 115, 118 first before running
python python 09_testing_and_dropout/03_parse_lrr_annotation_test.py

# we will also need to generate bandpass b-factor values to update the training model with

python 01_LRR_Annotation/analyze_bfactor_peaks.py \
  --pdb-dir ./09_testing_and_dropout/Ngou_2025_SCORE_data/pdb_for_lrr_annotator/ \
  -o ./09_testing_and_dropout/Ngou_2025_SCORE_data/bfactor_winding_lrr_segments.csv

# update lines 246, 263 with path for Ngou data
python python 01_LRR_Annotation/analyze_bfactor_peaks.py
```
We will then prepare our data in the formate for training and testing as well as add additional features. 

```
## Update line 90 and 108 and 116
python 09_testing_and_dropout/04_data_prep_for_test.py

## then in the 09_testing_and_dropout directory, run the following:
Rscript 05_chemical_conversion_test.R all Ngou_data_few_shot_seq.csv
```

For the Ngou data, we are going to compare zero-shot versus few-shot prediction. So we will run additional script to break the data in to train/test splits for few-shot. 

```
python 09_testing_and_dropout/06_few_shot_script.py
Created 05_datasets/few_shot_2 with 2 training samples
  - Class distribution: {'Immunogenic': 1, 'Weakly Immunogenic': 1}
Created 05_datasets/few_shot_4 with 4 training samples
  - Class distribution: {'Weakly Immunogenic': 3, 'Immunogenic': 1}
Created 05_datasets/few_shot_8 with 8 training samples
  - Class distribution: {'Weakly Immunogenic': 5, 'Immunogenic': 2, 'Non-Immunogenic': 1}
Created 05_datasets/few_shot_16 with 16 training samples
  - Class distribution: {'Non-Immunogenic': 8, 'Weakly Immunogenic': 6, 'Immunogenic': 2}
Created 05_datasets/few_shot_32 with 32 training samples
  - Class distribution: {'Non-Immunogenic': 21, 'Weakly Immunogenic': 7, 'Immunogenic': 4}
```

We will then update main train for these script and assess how mamp-ml improves. See 

## 06. How does mamp-ml compare to other structural models

Assessing how mamp-ml compares to other structural models is important for determining its potential advantages. There are many structural models that exist but we will test a couple to A) see how they perform and B) determine if the outputs of those models in combination with a simple regession model can outcompete mamp-ml performance. 

```
# make director of outputs 
mkdir 16_structural_models

# we then wrote a script that inputs All_LRR_PRR_ligand_data.xlsx and outputs an updated excel sheet that has co-receptor information as well (we will want to compare structural models with and without co-receptor information)

# install packages and run script
conda install -y -c bioconda -c conda-forge parasail-python
python 16_structural_models/serk_pipeline.py
```

Next, we need to build the multiple sequence alignments (MSAs) for each receptor sequence. This is an essential step for structural modeling, as high-quality MSAs will be used as input features for downstream structure prediction tools such as Boltz and will directly impact the accuracy of the resulting structural models.

```
# output for msas
mkdir -p 16_structural_models/receptor_only_msas

# install colabfold
pip install uv
uv venv ~/envs/colabfold --python 3.11
source ~/envs/colabfold/bin/activate
uv pip install colabfold

# MMseqs2 static binary for macOS (Homebrew's version can lag behind what ColabFold needs)
colabfold_search 03_out_data/lrr_domain_sequences.fasta ../colabfold_db 16_structural_models/receptor_only_msas \
  --mmseqs /opt/homebrew/bin/mmseqs --use-env 1 --use-templates 0 --threads 14
```

Next, we need to prepare the necessary input files in the specific formats required by each structural model we plan to use. This involves running scripts that convert our processed sequence and annotation data into model-specific input directories and file types to ensure compatibility for downstream structure prediction analyses.


```
python 16_structural_models/build_inputs.py \
  16_structural_models/SERK_coreceptor.tsv \
  03_out_data/lrr_domain_sequences.fasta \
  16_structural_models/receptor_only_msas \
  16_structural_models/input_files_fold

# outputs the following:
├── 16_structural_models/
│   └── input_files_fold/

```
First, we will install and run all receptor-epitope and receptor-epitope-co-receptor members from Model_Building section though Boltz-1. We will begin by running Boltz, a diffusion-based structural modeling tool, on our prepared input files. This will allow us to generate structural predictions for all receptor-epitope and receptor-epitope-co-receptor combinations, serving as a benchmark for our model comparison.

```
# --- Boltz-1 + Boltz-2 (same package, pick model with --model) -------------
# boltz-community is a maintained fork with native `--accelerator mps`.
uv venv envs/boltz --python 3.11
source envs/boltz/bin/activate
uv pip install boltz-community pandas pyyaml
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Boltz-1
boltz predict 16_structural_models/input_files_fold/boltz \
--model boltz1 \
--out_dir 16_structural_models/input_files_fold/boltz1 \
--accelerator mps \
--devices 1 \
--recycling_steps 1 \
--sampling_steps 100 \
--num_subsampled_msa 256 \
--diffusion_samples 1 \
--output_format mmcif

# Boltz-2
boltz predict 16_structural_models/input_files_fold/boltz \
--model boltz2 \
--out_dir 16_structural_models/input_files_fold/boltz2 \
--accelerator mps \
--devices 1 \
--recycling_steps 3 \
--sampling_steps 200 \
--diffusion_samples 1 \
--output_format mmcif
deactivate
```




```

# RF3
source ~/envs/rf3/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
rf3 fold inputs="$PWD/16_structural_models/fold_bench/inputs/rf3" out_dir="$PWD/16_structural_models/fold_bench/outputs/rf3" early_stopping_plddt_threshold=0.0 diffusion_batch_size=5 raise_if_missing_msa_for_protein_of_length_n=50 skip_existing=true
deactivate



rf3 fold inputs="$PWD/16_structural_models/fold_bench/rf3.json" out_dir="$PWD/16_structural_models/fold_bench/outputs/rf3" early_stopping_plddt_threshold=0.0 diffusion_batch_size=5 raise_if_missing_msa_for_protein_of_length_n=50 skip_existing=true

```






```

```


```

# --- RoseTTAFold3 (Foundry, Apple Silicon fork) -----------------------------
uv venv envs/rf3 --python 3.12
source envs/rf3/bin/activate
uv pip install torch pandas
uv pip install "rc-foundry[all] @ git+https://github.com/fnachon/foundry.git"
foundry install rf3          # downloads latest RF3 checkpoint to ~/.foundry/checkpoints
deactivate

source ~/envs/rf3/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1

rf3 fold inputs="$PWD/16_structural_models/fold_bench/inputs/rf3" \
  out_dir="$PWD/16_structural_models/fold_bench/outputs/rf3" \
  early_stopping_plddt_threshold=0.0 \
  diffusion_batch_size=5 \
  raise_if_missing_msa_for_protein_of_length_n=50 \
  skip_existing=true
deactivate
```
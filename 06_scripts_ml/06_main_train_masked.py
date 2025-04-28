# 06_scripts_ml/06_main_train_masked.py

#-----------------------------------------------------------------------------------------------
# Krasileva Lab - Plant & Microbial Biology Department UC Berkeley
# Author: Danielle M. Stevens (Original), Modified by AI Assistant
# Last Updated: [Current Date]
# Script Purpose: Train MAMP-receptor models with receptor masking based on B-factors.
# Inputs: Training/Testing data CSVs (must include 'protein_key'), B-factor CSV
# Outputs: Trained model checkpoints, evaluation metrics, WandB logs
#-----------------------------------------------------------------------------------------------

""""
  python 06_scripts_ml/06_main_train_masked.py \
        --model esm2_with_receptor_masked \
        --data_dir path/to/your/data_containing_protein_key_csvs \
        --bfactor_csv_path 04_Preprocessing_results/bfactor_winding_lrr_segments.csv \
        --epochs 50 \
        --batch_size 8 \
        # ... other arguments ...
"""


"""
Main training script for MAMP-receptor interaction models with receptor masking.

This script modifies the original training pipeline to incorporate masking
of specific receptor residues based on negative B-factor values from a CSV file.
It uses the ESMWithReceptorMaskedModel.

Requires input data CSVs to contain a 'protein_key' column for receptor mapping.
"""

import argparse
import datetime
import time
import os
import wandb
import random
import numpy as np
import pandas as pd
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

# random forest model
from models.random_forest_baseline import RandomForestBaselineModel

# esm models
from models.esm_model import ESMModel
from models.esm_mid_model import ESMMidModel
from models.esm_with_receptor_model import ESMWithReceptorModel
# ---> ADDED: Import the new masked model <---
from models.esm_with_receptor_masked_model import ESMWithReceptorMaskedModel
# ---------------------------------------------
from models.esm_receptor_chemical_fusion_variants import ESMReceptorChemical
from models.esm_with_receptor_single_seq_model import ESMWithReceptorSingleSeqModel
from models.esm_with_receptor_attn_film_model import ESMWithReceptorAttnFilmModel
from models.esm_contrast_model import ESMContrastiveModel

# glm models
from models.glm_model import GLMModel
from models.glm_with_receptor_model import GLMWithReceptorModel
from models.glm_with_receptor_single_seq_model import GLMWithReceptorSingleSeqModel

# alphafold and other models
from models.alphafold_model import AlphaFoldModel
from models.amp_with_receptor_model import AMPWithReceptorModel
from models.amp_model import AMPModel

from engine_train import train_one_epoch, evaluate
from datasets.seq_dataset import PeptideSeqDataset
from datasets.alphafold_dataset import AlphaFoldDataset
# ---> UPDATED: Import the original and the new masked dataset <---
from datasets.seq_with_receptor_dataset import PeptideSeqWithReceptorDataset
from datasets.seq_with_receptor_masked_dataset import PeptideSeqWithReceptorMaskedDataset # Import new dataset
# ----------------------------------------------------------------
import misc
from sklearn.model_selection import StratifiedKFold


def get_args_parser():
    parser = argparse.ArgumentParser("Train Sequence Detector with Receptor Masking") # Updated description
    # --- Basic Parameters ---
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--debug", default="", type=str)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        # ---> ADDED: Added the new model to choices <---
        choices=[
            "esm2", "glm2", "esm2_mid", "alphafold_pair_reps",
            "esm2_with_receptor", "esm_receptor_chemical", "glm2_with_receptor",
            "esm2_with_receptor_single_seq", "glm2_with_receptor_single_seq",
            "amplify", "amplify_with_receptor", "esm2_with_receptor_attn_film",
            "esm2_contrast", "random_forest", "esm2_with_receptor_masked" # New model added
        ]
        # ---------------------------------------------
    )
    parser.add_argument("--data_dir", type=str, required=False)
    parser.add_argument("--eval_only_data_path", type=str, required=False)

    # --- Model Parameters ---
    parser.add_argument("--backbone", default="esm2_t33_650M_UR50D", help="af|esm2_t33_650M_UR50D|esm_msa1b_t12_100M_UR50S")
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--freeze_at", type=int, default=14)
    parser.add_argument("--finetune_backbone", type=str, default="/scratch/cluster/jozhang/models/openfold_params/finetuning_ptm_2.pt")

    # --- Training Parameters ---
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-9)
    parser.add_argument("--weight_decay", type=float, default=0.5)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--cross_eval_kfold", type=int, help="Number of folds for cross-validation")

    # --- Data Parameters ---
    parser.add_argument("--max_context_length", type=int, default=2000)
    parser.add_argument("--num_workers", default=10, type=int)
    # ---> ADDED: Argument for B-factor CSV path <---
    parser.add_argument(
        "--bfactor_csv_path",
        type=str,
        default="04_Preprocessing_results/bfactor_winding_lrr_segments.csv",
        help="Path to the CSV file containing B-factors for receptor masking."
    )
    # ---------------------------------------------

    # --- Loss Parameters ---
    parser.add_argument("--lambda_single", type=float, default=0.1)
    parser.add_argument("--loss_single_aug", type=str, default="none")
    parser.add_argument("--lambda_double", type=float, default=1.0)
    parser.add_argument("--double_subsample_destabilizing_ratio", type=float, default=8)
    parser.add_argument("--lambda_pos", type=float, default=4)

    # --- Evaluation Parameters ---
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--dist_eval", action="store_true")
    parser.add_argument("--eval_reverse", action="store_true")
    parser.add_argument("--test", action="store_true")

    # --- Resume Parameters ---
    parser.add_argument("--finetune", default="", type=str)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--start_epoch", type=int, default=0)

    # --- Logging Parameters ---
    parser.add_argument("--save_pred_dict", action="store_true")
    parser.add_argument("--eval_period", type=int, default=5)
    parser.add_argument("--save_period", type=int, default=1000)
    parser.add_argument("--disable_wandb", action="store_true")
    # parser.add_argument("--wandb_group", default="krasileva", type=str) # Keep commented out if not needed
    parser.add_argument("--model_checkpoint_path", type=str, help="Path to model checkpoint for loading")

    # --- Distributed Training Parameters ---
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--name_to_x", default="../out_data/colabfold_name_to_x.pt", type=str)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")

    # --- Other Original Parameters (Keep as is unless needed) ---
    parser.add_argument("--repeat_thresh", type=float, default=0.001)
    parser.add_argument("--n_fed_cats", type=int, default=-1)
    parser.add_argument("--detic_path", default="/path/to/file", type=Path)
    parser.add_argument("--pos_thresh", type=float, default=0.5)
    parser.add_argument("--aa_expand", default="scratch", help="scratch|backbone")
    parser.add_argument("--single_dec", default="naive", help="naive|delta")
    parser.add_argument("--n_msa_seqs", type=int, default=128)
    parser.add_argument("--n_extra_msa_seqs", type=int, default=1024)
    parser.add_argument("--af_extract_feat", type=str, default="both", help="both|evo|struct")
    parser.add_argument("--contrastive_output", default=True)

    return parser


# ---> UPDATED: model_dict remains the same, pointing to the masked model class <---
model_dict = {
    "esm2": ESMModel,
    "glm2": GLMModel,
    "esm2_mid": ESMMidModel,
    "alphafold_pair_reps": AlphaFoldModel,
    "esm2_with_receptor": ESMWithReceptorModel,
    "esm_receptor_chemical": ESMReceptorChemical,
    "glm2_with_receptor": GLMWithReceptorModel,
    "esm2_with_receptor_single_seq": ESMWithReceptorSingleSeqModel,
    "glm2_with_receptor_single_seq": GLMWithReceptorSingleSeqModel,
    "amplify": AMPModel,
    "amplify_with_receptor": AMPWithReceptorModel,
    "esm2_with_receptor_attn_film": ESMWithReceptorAttnFilmModel,
    "esm2_contrast": ESMContrastiveModel,
    "random_forest": RandomForestBaselineModel,
    "esm2_with_receptor_masked": ESMWithReceptorMaskedModel # Correct model class
}
# ---------------------------------------------

# ---> UPDATED: Map the masked model name to the new dataset class <---
dataset_dict = {
    "esm2": PeptideSeqDataset,
    "glm2": PeptideSeqDataset,
    "esm2_mid": PeptideSeqDataset,
    "alphafold_pair_reps": AlphaFoldDataset,
    "esm2_with_receptor": PeptideSeqWithReceptorDataset, # Original uses original dataset
    "esm_receptor_chemical": PeptideSeqWithReceptorDataset,
    "glm2_with_receptor": PeptideSeqWithReceptorDataset,
    "esm2_with_receptor_single_seq": PeptideSeqWithReceptorDataset,
    "glm2_with_receptor_single_seq": PeptideSeqWithReceptorDataset,
    "amplify": PeptideSeqDataset,
    "amplify_with_receptor": PeptideSeqWithReceptorDataset,
    "esm2_with_receptor_attn_film": PeptideSeqWithReceptorDataset,
    "esm2_contrast": PeptideSeqDataset,
    "random_forest": PeptideSeqWithReceptorDataset,
    "esm2_with_receptor_masked": PeptideSeqWithReceptorMaskedDataset # Use the new masked dataset class
}
# ---------------------------------------------

# ---> UPDATED: wandb_dict remains the same <---
wandb_dict = {
    "esm2": "mamp_esm2",
    "glm2": "mamp_glm2",
    "esm2_mid": "mamp_esm2_mid",
    "alphafold_pair_reps": "mamp_alphafold_pair_reps",
    "esm_receptor_chemical": "mamp_esm_receptor_chemical",
    "esm2_with_receptor": "mamp_esm2_with_receptor",
    "glm2_with_receptor": "mamp_glm2_with_receptor",
    "esm2_with_receptor_single_seq": "mamp_esm2_with_receptor_single_seq",
    "glm2_with_receptor_single_seq": "mamp_glm2_with_receptor_single_seq",
    "amplify": "mamp_amplify",
    "amplify_with_receptor": "mamp_amplify_with_receptor",
    "esm2_with_receptor_attn_film": "mamp_esm2_with_receptor_attn_film",
    "esm2_contrast": "mamp_esm2_contrast",
    "random_forest": "mamp_random_forest",
    "esm2_with_receptor_masked": "mamp_esm2_with_receptor_masked"
}
# ---------------------------------------------


def main(args):
    """Main training/evaluation function adapted for receptor masking."""
    misc.init_distributed_mode(args)

    # Setup WandB logging
    if not args.disable_wandb and misc.is_main_process():
        current_datetime = datetime.datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
        # Use the specific wandb name from the dictionary
        wandb_run_base_name = wandb_dict.get(args.model, f"mamp_{args.model}") # Fallback if model not in dict
        if args.eval_only_data_path:
            run_name = f"{wandb_run_base_name}-{Path(args.eval_only_data_path).stem}-{current_datetime}"
            tags = [args.model, str(Path(args.eval_only_data_path).stem), "eval"]
        else:
            data_name = Path(args.data_dir).name if args.data_dir else "unknown_data"
            run_name = f"{wandb_run_base_name}-{data_name}-{current_datetime}"
            tags = [args.model, data_name, "train"]
        wandb.init(
            project="mamp_ml",
            entity="dmstev-uc-berkeley", # Replace if needed
            name=run_name,
            config=args,
            dir=args.output_dir,
            tags=tags
        )
    print(args)

    device = torch.device(args.device)

    # Set random seeds
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ---> Model Initialization remains the same <---
    model_cls = model_dict[args.model]
    model_kwargs = {'args': args}
    if args.model == "esm2_with_receptor_masked":
        model_kwargs['bfactor_csv_path'] = args.bfactor_csv_path
        print(f"Initializing {args.model} with B-factor masking from: {args.bfactor_csv_path}")
    model = model_cls(**model_kwargs)
    # ---------------------------------------------

    # Load checkpoint if specified
    if args.model_checkpoint_path:
        # Make sure map_location is set correctly if loading from a different device
        map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank} if args.distributed else args.device
        print(f"Loading model checkpoint from: {args.model_checkpoint_path}")
        checkpoint = torch.load(args.model_checkpoint_path, map_location=map_location)

        # Adjust state dict loading based on potential DDP saving
        state_dict = checkpoint.get('model', checkpoint) # Handle different checkpoint formats
        # Handle potential 'module.' prefix if saved from DDP
        if all(k.startswith('module.') for k in state_dict.keys()):
             state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
        elif any(k.startswith('module.') for k in state_dict.keys()):
             print("Warning: Mixed keys ('module.' prefix) found in checkpoint state_dict. Attempting load.")
             state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        load_result = model.load_state_dict(state_dict, strict=False) # Use strict=False initially for flexibility
        print(f"Model load result: Missing keys: {load_result.missing_keys}, Unexpected keys: {load_result.unexpected_keys}")
        if load_result.missing_keys or load_result.unexpected_keys:
             print("Warning: Mismatched keys found during checkpoint loading. Check model definition and checkpoint.")


    # ---> Dataset Initialization uses the updated dataset_dict <---
    dataset_cls = dataset_dict[args.model]
    dataset_kwargs = {}
    if issubclass(dataset_cls, AlphaFoldDataset):
         try:
             name_to_x_data = torch.load(args.name_to_x)
             dataset_kwargs['name_to_x'] = name_to_x_data
         except FileNotFoundError:
             print(f"Error: AlphaFold name_to_x file not found at {args.name_to_x}")
             exit(1)
    # -----------------------------------------------------------

    # Print model statistics
    n_params = sum(p.numel() for p in model.parameters())
    n_params_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model) # Print model architecture
    if args.eval_only_data_path:
        print(f"Evaluating a model with {n_params_grad:,} trainable parameters out of {n_params:,} parameters")
    else:
        print(f"Training {n_params_grad:,} of {n_params:,} parameters")


    # Get model's collate function
    # ---> IMPORTANT: Ensure the model instance provides the collate_fn <---
    if hasattr(model, 'collate_fn') and callable(model.collate_fn):
        collate_fn = model.collate_fn
    else:
        # Fallback or error if the model doesn't provide it
        print(f"Warning: Model class {model_cls.__name__} does not have a 'collate_fn'. Using default Pytorch collate (might fail).")
        from torch.utils.data.dataloader import default_collate
        collate_fn = default_collate
    # --------------------------------------------------------------------

    model.to(args.device)
    model_without_ddp = model
    if args.distributed:
        # Ensure correct device_ids mapping
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, # Specify local_rank
            find_unused_parameters=True # Set True if expect unused params (e.g. due to conditional logic)
        )
        model_without_ddp = model.module
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
    else: # Ensure these are defined even in non-distributed mode
        num_tasks = 1
        global_rank = 0


    # Initialize optimizer
    # Consider excluding non-trainable parameters if freezing parts of the model
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # param_groups = misc.param_groups_weight_decay(model, args.weight_decay) # This might include non-trainable params
    param_groups = misc.param_groups_weight_decay(
        nn.ModuleList(trainable_params), # Pass only trainable params here if using custom grouping
        args.weight_decay
    ) if hasattr(misc, 'param_groups_weight_decay') else trainable_params # Fallback if misc func is complex

    # Handle case where param_groups might be empty (if all frozen)
    if not param_groups:
         print("Warning: No trainable parameters found. Optimizer will not be created.")
         optimizer = None
    else:
         optimizer = optim.AdamW(param_groups, lr=args.lr) # Removed weight decay here if handled by param_groups
         print(f"Optimizer: AdamW, LR={args.lr}, Weight Decay (if applicable in param_groups): {args.weight_decay}")


    # Load optimizer state if resuming training (and optimizer exists)
    if optimizer:
         # Pass optimizer to load_model (assuming it handles scheduler=None)
         misc.load_model(args, model_without_ddp, optimizer, None)
    else:
         # Still potentially load model state even if optimizer isn't used (e.g., for eval)
         misc.load_model(args, model_without_ddp, None, None)


    # --- Prepare Test Data ---
    if args.eval_only_data_path:
        eval_data_path = args.eval_only_data_path
    elif args.data_dir:
        eval_data_path = f"{args.data_dir}/test_data_with_bulkiness.csv" # Or test_stratify.csv
        # eval_data_path = f"{args.data_dir}/test_stratify.csv"
    else:
        print("Error: Either --eval_only_data_path or --data_dir must be provided for evaluation data.")
        exit(1)

    print(f"Loading evaluation data from: {eval_data_path}")
    try:
        test_df = pd.read_csv(eval_data_path)
        # ---> Check if 'Header_Name' exists if using the masked model's dataset <---
        if dataset_cls == PeptideSeqWithReceptorMaskedDataset and 'Header_Name' not in test_df.columns:
             print(f"Error: Evaluation CSV '{eval_data_path}' must contain a 'Header_Name' column for model '{args.model}'.")
             exit(1)
        # -------------------------------------------------------------------------
        ds_test = dataset_cls(df=test_df, **dataset_kwargs) # Pass kwargs here
        print(f"Test dataset size: {len(ds_test)}")
    except FileNotFoundError:
        print(f"Error: Evaluation data file not found at {eval_data_path}")
        exit(1)
    except Exception as e:
        print(f"Error loading evaluation data: {e}")
        exit(1)


    if args.distributed and args.dist_eval:
        # DDP evaluation sampler
        if len(ds_test) % num_tasks != 0:
            print(f"Warning: Test set size ({len(ds_test)}) not divisible by world size ({num_tasks}). Some samples might be duplicated.")
        sampler_test = torch.utils.data.DistributedSampler(
            ds_test, num_replicas=num_tasks, rank=global_rank, shuffle=False # Usually False for eval
        )
    else:
        # Standard sequential sampler for evaluation
        sampler_test = torch.utils.data.SequentialSampler(ds_test)

    dl_test = torch.utils.data.DataLoader(
        ds_test,
        sampler=sampler_test,
        batch_size=args.batch_size, # Consider a larger batch size for evaluation if memory allows
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True, # Improves data transfer speed to GPU
        drop_last=False # Keep all evaluation samples
    )

    # --- Evaluation Only Mode ---
    if args.eval or args.eval_only_data_path: # Combine flags for clarity
        print("Running in evaluation-only mode.")
        # Ensure model is in eval mode
        model.eval()
        metrics = {}
        # Use torch.no_grad() for efficiency during evaluation
        with torch.no_grad():
             metrics.update(evaluate(model, dl_test, device, args, args.output_dir))
        print("Evaluation metrics:", metrics)
        if not args.disable_wandb and misc.is_main_process():
            # Log final eval metrics to wandb
            wandb.log(metrics) # Log the dictionary directly
            wandb.finish()
        exit() # Exit after evaluation

    # --- Prepare Training Data (Only if not in eval mode) ---
    if not args.data_dir:
        print("Error: --data_dir must be provided for training.")
        exit(1)

    train_data_path = f"{args.data_dir}/train_data_with_bulkiness.csv" # Or train_stratify.csv
    # train_data_path = f"{args.data_dir}/train_stratify.csv"
    print(f"Loading training data from: {train_data_path}")
    try:
        train_df = pd.read_csv(train_data_path)
        # ---> Check if 'Header_Name' exists if using the masked model's dataset <---
        if dataset_cls == PeptideSeqWithReceptorMaskedDataset and 'Header_Name' not in train_df.columns:
             print(f"Error: Training CSV '{train_data_path}' must contain a 'Header_Name' column for model '{args.model}'.")
             exit(1)
        # --------------------------------------------------------------------------
        ds_train = dataset_cls(df=train_df, **dataset_kwargs) # Pass kwargs here
        print(f"Train dataset size: {len(ds_train)}")
    except FileNotFoundError:
        print(f"Error: Training data file not found at {train_data_path}")
        exit(1)
    except Exception as e:
        print(f"Error loading training data: {e}")
        exit(1)


    if args.distributed:
        # DDP training sampler (shuffled)
        sampler_train = torch.utils.data.DistributedSampler(
            ds_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed
        )
        print(f"Using DistributedSampler for training (shuffle=True, seed={args.seed}).")
    else:
        # Standard random sampler for training
        sampler_train = torch.utils.data.RandomSampler(ds_train)
        print("Using RandomSampler for training.")

    dl_train = torch.utils.data.DataLoader(
        ds_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True, # Generally good for training speed
        drop_last=True   # Drop last incomplete batch during training
    )

    # --- Training Loop ---
    print(f"Start training for {args.epochs} epochs, saving to {args.output_dir}")
    start_time = time.time()

    # Initial evaluation before training starts
    args.current_epoch = 0
    print("Running initial evaluation before training...")
    model.eval() # Set model to eval mode for evaluation
    with torch.no_grad(): # Disable gradients for evaluation
         initial_metrics = evaluate(model, dl_test, device, args, args.output_dir)
    print("Initial evaluation metrics:", initial_metrics)
    # Log initial metrics to WandB if enabled
    if not args.disable_wandb and misc.is_main_process():
         wandb.log({"epoch": 0, **initial_metrics}) # Log with epoch 0


    for epoch in range(args.start_epoch, args.epochs):
        args.current_epoch = epoch + 1 # Use 1-based epoch for logging? Or keep 0-based? Let's use 1-based for user display.
        print(f"\n--- Epoch {args.current_epoch}/{args.epochs} ---")

        if args.distributed:
            # Set epoch for sampler to ensure proper shuffling
            sampler_train.set_epoch(epoch)

        # Train one epoch
        model.train() # Set model to train mode
        train_stats = train_one_epoch(model, dl_train, optimizer, device, epoch, args) # Pass epoch (0-based)

        # Log training stats
        if not args.disable_wandb and misc.is_main_process():
             # Log training stats with learning rate
             log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                          'epoch': args.current_epoch, # Log 1-based epoch
                          'lr': optimizer.param_groups[0]['lr'] if optimizer else 0}
             wandb.log(log_stats)


        # Periodic evaluation
        if (epoch + 1) % args.eval_period == 0 or (epoch + 1) == args.epochs: # Eval on eval_period intervals and final epoch
            print(f"--- Evaluating at Epoch {args.current_epoch} ---")
            model.eval() # Set model to eval mode
            with torch.no_grad(): # Disable gradients
                 eval_metrics = evaluate(model, dl_test, device, args, args.output_dir)
            print(f"Evaluation metrics at epoch {args.current_epoch}:", eval_metrics)

            # Log evaluation metrics
            if not args.disable_wandb and misc.is_main_process():
                 wandb.log({"epoch": args.current_epoch, **eval_metrics}) # Log with 1-based epoch


        # Periodic checkpointing (save based on 0-based epoch)
        # Save based on epoch number (e.g., after epoch 9 for epoch 10 done)
        if (epoch + 1) % args.save_period == 0 or (epoch + 1) == args.epochs:
             if misc.is_main_process(): # Only save on main process
                 print(f"--- Saving checkpoint at Epoch {args.current_epoch} ---")
                 ckpt_path = misc.save_model(
                     args=args,
                     epoch=epoch, # Save with the completed 0-based epoch number
                     model=model, # Pass DDP model if distributed
                     model_without_ddp=model_without_ddp,
                     optimizer=optimizer,
                     loss_scaler=None # Assuming no loss scaler for now
                 )
                 print(f"Saved checkpoint to {ckpt_path}")
                 # Optionally save to WandB as artifact
                 # if not args.disable_wandb:
                 #     artifact = wandb.Artifact(f'model-epoch-{epoch+1}', type='model')
                 #     artifact.add_file(ckpt_path)
                 #     wandb.log_artifact(artifact)

    # --- End of Training ---
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")

    # Final evaluation is already done within the loop's last iteration if condition `(epoch + 1) == args.epochs` is met.

    # Finish WandB run
    if not args.disable_wandb and misc.is_main_process():
        wandb.finish()

    # --- Optional k-fold cross-validation ---
    if args.cross_eval_kfold:
        print(f"\n--- Starting {args.cross_eval_kfold}-Fold Cross-Validation ---")
        if not args.data_dir:
            print("Error: --data_dir must be provided for cross-validation.")
            exit(1)
        full_data_path = f"{args.data_dir}/train_data_with_bulkiness.csv" # Adjust if needed
        print(f"Loading full data for cross-validation from: {full_data_path}")
        try:
             full_df = pd.read_csv(full_data_path)
             # ---> Check for 'Header_Name' if using masked model's dataset <---
             if dataset_cls == PeptideSeqWithReceptorMaskedDataset and 'Header_Name' not in full_df.columns:
                 print(f"Error: Cross-validation CSV '{full_data_path}' must contain 'Header_Name'.")
                 exit(1)
             # ------------------------------------------------------------------
             stratify_col = 'y' # Assuming 'y' is the label for stratification. Adjust if needed.
             if stratify_col not in full_df.columns:
                 # Try mapping 'Known Outcome' first if 'y' isn't directly present
                 if 'Known Outcome' in full_df.columns:
                     full_df['y'] = full_df['Known Outcome'].map(category_to_index).fillna(-1).astype(int)
                     print(f"Using mapped 'Known Outcome' as stratification column 'y'.")
                 else:
                     print(f"Error: Stratification column '{stratify_col}' (or 'Known Outcome') not found in {full_data_path}.")
                     exit(1)

        except FileNotFoundError:
             print(f"Error: Full data file not found at {full_data_path}")
             exit(1)
        except Exception as e:
             print(f"Error loading full data for CV: {e}")
             exit(1)

        skf = StratifiedKFold(n_splits=args.cross_eval_kfold, random_state=args.seed, shuffle=True)
        fold_metrics = []

        for i, (train_idx, test_idx) in enumerate(skf.split(full_df, full_df[stratify_col])):
            print(f"\n--- Cross-Validation Fold {i+1}/{args.cross_eval_kfold} ---")

            # Initialize WandB for this fold if enabled
            if not args.disable_wandb and misc.is_main_process():
                 # Use a consistent base name and add fold number
                 wandb_run_base_name = wandb_dict.get(args.model, f"mamp_{args.model}")
                 data_name = Path(args.data_dir).name if args.data_dir else "unknown_data"
                 cv_run_name = f"{wandb_run_base_name}-{data_name}-CV_Fold_{i+1}"
                 wandb.init(
                     project="mamp_ml_cv", # Consider a separate project or group for CV runs
                     entity="dmstev-uc-berkeley", # Replace if needed
                     name=cv_run_name,
                     config=args,
                     dir=args.output_dir,
                     tags=[args.model, data_name, "train", f"cv_fold_{i+1}"],
                     reinit=True # Allow reinitialization for each fold
                 )

            # Re-initialize model for each fold
            cv_model_cls = model_dict[args.model]
            cv_model_kwargs = {'args': args}
            if args.model == "esm2_with_receptor_masked":
                 cv_model_kwargs['bfactor_csv_path'] = args.bfactor_csv_path
            cv_model = cv_model_cls(**cv_model_kwargs)
            cv_model.to(args.device)

            cv_model_without_ddp = cv_model
            if args.distributed:
                 # Re-wrap model in DDP for each fold
                 cv_model = torch.nn.parallel.DistributedDataParallel(
                     cv_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
                 )
                 cv_model_without_ddp = cv_model.module

            # Re-initialize optimizer for each fold
            cv_trainable_params = [p for p in cv_model.parameters() if p.requires_grad]
            cv_param_groups = misc.param_groups_weight_decay(
                 nn.ModuleList(cv_trainable_params), args.weight_decay
            ) if hasattr(misc, 'param_groups_weight_decay') else cv_trainable_params

            if not cv_param_groups:
                 cv_optimizer = None
            else:
                 cv_optimizer = optim.AdamW(cv_param_groups, lr=args.lr)

            # No need to load checkpoints usually for CV, train from scratch per fold
            # misc.load_model(args, cv_model_without_ddp, cv_optimizer, None) # Skip unless intended

            # Create datasets for this fold using the correct dataset class
            cv_df_train = full_df.iloc[train_idx]
            cv_df_test = full_df.iloc[test_idx]
            # ---> Use dataset_cls determined earlier <---
            cv_ds_train = dataset_cls(df=cv_df_train, **dataset_kwargs)
            cv_ds_test = dataset_cls(df=cv_df_test, **dataset_kwargs)
            # ------------------------------------------

            # Create dataloaders for this fold
            if args.distributed:
                 cv_sampler_train = torch.utils.data.DistributedSampler(
                     cv_ds_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed + i # Seed per fold
                 )
                 cv_sampler_test = torch.utils.data.DistributedSampler(
                     cv_ds_test, num_replicas=num_tasks, rank=global_rank, shuffle=False
                 )
            else:
                 cv_sampler_train = torch.utils.data.RandomSampler(cv_ds_train)
                 cv_sampler_test = torch.utils.data.SequentialSampler(cv_ds_test)

            cv_dl_train = torch.utils.data.DataLoader(
                 cv_ds_train, sampler=cv_sampler_train, batch_size=args.batch_size,
                 collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True, drop_last=True
            )
            cv_dl_test = torch.utils.data.DataLoader(
                 cv_ds_test, sampler=cv_sampler_test, batch_size=args.batch_size, # Use training batch size or different?
                 collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True, drop_last=False
            )

            # Training loop for this fold
            print(f"Starting training for fold {i+1}...")
            for epoch in range(args.start_epoch, args.epochs): # Use 0-based epoch internally
                 cv_current_epoch = epoch + 1
                 if args.distributed:
                     cv_sampler_train.set_epoch(epoch)

                 cv_model.train()
                 cv_train_stats = train_one_epoch(cv_model, cv_dl_train, cv_optimizer, device, epoch, args)

                 # Log CV train stats per fold
                 if not args.disable_wandb and misc.is_main_process():
                      log_cv_train = {**{f'train_{k}': v for k, v in cv_train_stats.items()},
                                      'epoch': cv_current_epoch,
                                      'lr': cv_optimizer.param_groups[0]['lr'] if cv_optimizer else 0,
                                      'fold': i + 1}
                      wandb.log(log_cv_train)

                 # Periodic evaluation within the fold (optional, can just eval at end)
                 # if (epoch + 1) % args.eval_period == 0:
                 #     cv_model.eval()
                 #     with torch.no_grad():
                 #          cv_eval_metrics = evaluate(cv_model, cv_dl_test, device, args, args.output_dir, prefix=f"fold_{i+1}_eval")
                 #     print(f"Fold {i+1} Eval metrics at epoch {cv_current_epoch}:", cv_eval_metrics)
                 #     if not args.disable_wandb and misc.is_main_process():
                 #          wandb.log({"epoch": cv_current_epoch, 'fold': i + 1, **cv_eval_metrics})


            # Final evaluation for this fold
            print(f"--- Evaluating Fold {i+1} after {args.epochs} epochs ---")
            cv_model.eval()
            with torch.no_grad():
                 final_fold_metrics = evaluate(cv_model, cv_dl_test, device, args, args.output_dir, prefix=f"fold_{i+1}_final")
            print(f"Final metrics for Fold {i+1}:", final_fold_metrics)
            fold_metrics.append(final_fold_metrics)

            # Log final fold metrics
            if not args.disable_wandb and misc.is_main_process():
                 wandb.log({"epoch": args.epochs, 'fold': i + 1, **final_fold_metrics})
                 # Finish WandB run for this fold
                 wandb.finish()

        # --- Aggregate and Print CV Results ---
        if misc.is_main_process(): # Only aggregate and print on main process
             print("\n--- Cross-Validation Summary ---")
             if fold_metrics:
                 # Example: Average accuracy across folds
                 avg_metrics = {}
                 all_keys = set(k for metrics in fold_metrics for k in metrics.keys())
                 for key in all_keys:
                     # Ensure key exists and value is numeric before averaging
                     valid_values = [metrics.get(key) for metrics in fold_metrics if isinstance(metrics.get(key), (int, float))]
                     if valid_values:
                          avg_metrics[f"avg_{key}"] = np.mean(valid_values)
                          avg_metrics[f"std_{key}"] = np.std(valid_values)

                 print("Average Metrics Across Folds:")
                 for k, v in avg_metrics.items():
                     print(f"  {k}: {v:.4f}")

                 # Optionally log aggregated CV results to a final WandB run or print to file
                 # Consider creating a final summary WandB run here if desired
             else:
                 print("No metrics collected during cross-validation.")


if __name__ == "__main__":
    args_parser = get_args_parser()
    args = args_parser.parse_args()

    # Define output directory based on mode and model
    timestamp = datetime.datetime.now().strftime("%m%d%Y_%H%M") # Use timestamp for uniqueness
    mode_prefix = "eval" if (args.eval or args.eval_only_data_path) else "train"
    data_name_part = Path(args.eval_only_data_path).stem if args.eval_only_data_path else (Path(args.data_dir).name if args.data_dir else "unknown_data")
    run_name = f"{args.model}_{data_name_part}_{mode_prefix}_{timestamp}"

    # Place results in a structured directory
    results_base_dir = Path("../eval_model_results") if mode_prefix == "eval" else Path("../model_results")
    out_dir = results_base_dir / run_name

    # Create output directory if it doesn't exist
    out_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = out_dir
    print(f"Output directory set to: {args.output_dir}")

    # Check critical requirements early
    # ---> Use dataset_dict to check which dataset class is being used <---
    selected_dataset_cls = dataset_dict.get(args.model) # Get the class itself
    if selected_dataset_cls == PeptideSeqWithReceptorMaskedDataset:
         if not args.bfactor_csv_path:
             print("Error: --bfactor_csv_path is required when using model 'esm2_with_receptor_masked'")
             exit(1)
         if not Path(args.bfactor_csv_path).is_file():
             print(f"Error: B-factor CSV file not found at {args.bfactor_csv_path}")
             exit(1)
    # ----------------------------------------------------------------------

    main(args)
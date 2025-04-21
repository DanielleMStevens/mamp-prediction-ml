#-----------------------------------------------------------------------------------------------
# Krasileva Lab - Plant & Microbial Biology Department UC Berkeley
# Author: Danielle M. Stevens
# Last Updated: 07/06/2020
# Script Purpose: 
# Inputs: 
# Outputs: 
#-----------------------------------------------------------------------------------------------

"""
Main training script for MAMP (Microbe-Associated Molecular Pattern)-receptor interaction models.

This script provides a comprehensive training pipeline for various deep learning models
designed to predict MAMP-receptor interactions. It supports multiple model architectures,
distributed training, and extensive evaluation capabilities.

Key Features:
- Multiple model architectures (ESM, GLM, AlphaFold, etc.)
- Distributed training support
- Cross-validation capabilities
- Extensive logging and visualization with WandB
- Checkpoint saving and loading
- Evaluation-only mode for model testing

The script is organized into several main components:
1. Argument parsing and configuration
2. Model initialization and setup
3. Data loading and preprocessing
4. Training loop
5. Evaluation
6. Cross-validation (optional)

Example Usage:
    # Training mode:
    python 05_main_train.py --model esm2 --data_dir path/to/data --epochs 50
    
    # Evaluation mode:
    python 06_scripts_ml/06_main_train.py --model esm2_with_receptor --eval_only_data_path 05_datasets/test_stratify.csv /
    --model_checkpoint_path ../model_results/02_24_2025_esm2_with_receptor_stratify/test_preds.pth --disable_wandb
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

# esm models
from models.esm_model import ESMModel
from models.esm_mid_model import ESMMidModel
from models.esm_with_receptor_model import ESMWithReceptorModel
#from models.esm_receptor_chemical import ESMReceptorChemical
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
from datasets.seq_with_receptor_dataset import PeptideSeqWithReceptorDataset
import misc
from sklearn.model_selection import StratifiedKFold



def get_args_parser():
    """
    Creates and returns an argument parser for the training script.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser with the following groups of parameters:
        
        Basic Parameters:
            --seed: Random seed for reproducibility
            --debug: Debug mode configuration
            --model: Model architecture to use (required)
            --data_dir: Directory containing training data
            --eval_only_data_path: Path to evaluation data (for eval mode)
            
        Model Parameters:
            --backbone: Choice of backbone model (default: esm2_t33_650M_UR50D)
            --head_dim: Dimension of the model head (default: 128)
            --freeze_at: Layer to freeze up to (default: 14)
            --finetune_backbone: Path to pretrained backbone weights
            
        Training Parameters:
            --epochs: Number of training epochs (default: 50)
            --batch_size: Batch size for training (default: 8)
            --lr: Learning rate (default: 3e-4)
            --min_lr: Minimum learning rate (default: 1e-9)
            --weight_decay: Weight decay for optimization (default: 0.5)
            --warmup_epochs: Number of warmup epochs (default: 10)
            
        Loss Parameters:
            --lambda_single: Weight for single sequence loss (default: 0.1)
            --lambda_double: Weight for double sequence loss (default: 1.0)
            --lambda_pos: Weight for positive examples (default: 4)
            
        Evaluation Parameters:
            --eval: Enable evaluation mode
            --dist_eval: Enable distributed evaluation
            --eval_period: Epochs between evaluations (default: 10)
            --save_period: Epochs between checkpoints (default: 1000)
            
        Logging Parameters:
            --disable_wandb: Disable WandB logging
            --wandb_group: WandB group name (default: krasileva)
            --save_pred_dict: Save prediction dictionary
            
        Distributed Training Parameters:
            --device: Device to use (default: cuda)
            --world_size: Number of distributed processes (default: 1)
            --dist_url: URL for distributed training
    """
    parser = argparse.ArgumentParser("Train Sequence Detector")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--debug", default="", type=str)

    # long_tail params
    parser.add_argument("--repeat_thresh", type=float, default=0.001)
    parser.add_argument("--n_fed_cats", type=int, default=-1)

    # model params
    parser.add_argument(
        "--detic_path",
        default="/path/to/file",
        type=Path,
        help="path to weights of linear head",
    )
    parser.add_argument("--pos_thresh", type=float, default=0.5)
    parser.add_argument("--aa_expand", default="scratch", help="scratch|backbone")
    parser.add_argument("--single_dec", default="naive", help="naive|delta")
    # parser.add_argument("ulti_dec", default="epistasis", help="additive|epistasis")
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--backbone", default="esm2_t33_650M_UR50D", help="af|esm2_t33_650M_UR50D|esm_msa1b_t12_100M_UR50S",
    )
    parser.add_argument(
        "--finetune_backbone",
        type=str,
        default="/scratch/cluster/jozhang/models/openfold_params/finetuning_ptm_2.pt",
    )
    parser.add_argument("--freeze_at", type=int, default=14)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=False)
    parser.add_argument("--eval_only_data_path", type=str, required=False)

    # af params
    parser.add_argument("--n_msa_seqs", type=int, default=128)
    parser.add_argument("--n_extra_msa_seqs", type=int, default=1024)
    parser.add_argument(
        "--af_extract_feat", type=str, default="both", help="both|evo|struct"
    )

    # Data parameters
    parser.add_argument("--max_context_length", type=int, default=2000,
                       help="Maximum context length for sequences")
    parser.add_argument("--num_workers", default=10, type=int,
                       help="Number of data loading workers")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Training batch size")
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-9,
                       help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.5,
                       help="Weight decay for optimization")
    parser.add_argument("--warmup_epochs", type=int, default=10,
                       help="Number of warmup epochs")
    parser.add_argument("--cross_eval_kfold", type=int,
                       help="Number of folds for cross-validation")

    # Loss parameters
    parser.add_argument("--lambda_single", type=float, default=0.1,
                       help="Weight for single sequence loss")
    parser.add_argument("--loss_single_aug", type=str, default="none",
                       help="Single loss augmentation (none|tp|forrev)")
    parser.add_argument("--lambda_double", type=float, default=1.0,
                       help="Weight for double sequence loss")
    parser.add_argument("--double_subsample_destabilizing_ratio", type=float, default=8,
                       help="Ratio for subsampling destabilizing pairs")
    parser.add_argument("--lambda_pos", type=float, default=4,
                       help="Weight for positive examples")

    # Evaluation parameters
    parser.add_argument("--eval", action="store_true",
                       help="Run in evaluation mode")
    parser.add_argument("--dist_eval", action="store_true",
                       help="Enable distributed evaluation")
    parser.add_argument("--eval_reverse", action="store_true",
                       help="Evaluate on reverse sequences")
    parser.add_argument("--test", action="store_true",
                       help="Use data_path instead of eval_data_paths")

    # Resume parameters
    parser.add_argument("--finetune", default="", type=str,
                       help="Path to finetune from")
    parser.add_argument("--resume", default="", type=str,
                       help="Path to resume from")
    parser.add_argument("--start_epoch", type=int, default=0,
                       help="Starting epoch number")

    # Logging parameters
    parser.add_argument("--save_pred_dict", action="store_true",
                       help="Save prediction dictionary")
    parser.add_argument("--eval_period", type=int, default=10,
                       help="Epochs between evaluations")
    parser.add_argument("--save_period", type=int, default=1000,
                       help="Epochs between checkpoints")
    parser.add_argument("--disable_wandb", action="store_true",
                       help="Disable WandB logging")
#    parser.add_argument("--wandb_group", default="krasileva", type=str,
#                       help="WandB group name")
    parser.add_argument("--model_checkpoint_path", type=str,
                       help="Path to model checkpoint for loading")

    # distributed training parameters
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--name_to_x", default="../out_data/colabfold_name_to_x.pt", type=str, help="path of name_to_x"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    parser.add_argument("--contrastive_output", default=True)
    return parser


# Dictionary mapping model names to their implementations
model_dict = {
    "esm2": ESMModel,                                      # ESM2 base model
    "glm2": GLMModel,                                      # GLM2 base model
    "esm2_mid": ESMMidModel,                              # ESM2 with mid-layer features
    "alphafold_pair_reps": AlphaFoldModel,                # AlphaFold-based model
    "esm2_with_receptor": ESMWithReceptorModel,           # ESM2 with receptor interaction
    #"esm_receptor_chemical": ESMReceptorChemical,            # ESM2 with chemical interaction
    "glm2_with_receptor": GLMWithReceptorModel,           # GLM2 with receptor interaction
    "esm2_with_receptor_single_seq": ESMWithReceptorSingleSeqModel,  # ESM2 with single sequence receptor
    "glm2_with_receptor_single_seq": GLMWithReceptorSingleSeqModel,  # GLM2 with single sequence receptor
    "amplify": AMPModel,                                  # AMP (Antimicrobial Peptide) model
    "amplify_with_receptor": AMPWithReceptorModel,        # AMP model with receptor
    "esm2_with_receptor_attn_film": ESMWithReceptorAttnFilmModel,  # ESM2 with attention and FiLM
    "esm2_contrast": ESMContrastiveModel                  # ESM2 with contrastive learning
}

# Dictionary mapping model names to their corresponding dataset classes
dataset_dict = {
    "esm2": PeptideSeqDataset,                           # Basic peptide sequence dataset
    "glm2": PeptideSeqDataset,
    "esm2_mid": PeptideSeqDataset,
    "alphafold_pair_reps": AlphaFoldDataset,             # Dataset for AlphaFold features
    "esm2_with_receptor": PeptideSeqWithReceptorDataset, # Dataset with receptor information
    "esm_receptor_chemical": PeptideSeqWithReceptorDataset, # Dataset with receptor information
    "glm2_with_receptor": PeptideSeqWithReceptorDataset,
    "esm2_with_receptor_single_seq": PeptideSeqWithReceptorDataset,
    "glm2_with_receptor_single_seq": PeptideSeqWithReceptorDataset,
    "amplify": PeptideSeqDataset,
    "amplify_with_receptor": PeptideSeqWithReceptorDataset,
    "esm2_with_receptor_attn_film": PeptideSeqWithReceptorDataset,
    "esm2_contrast": PeptideSeqDataset
}

# Dictionary mapping model names to their WandB experiment names
wandb_dict = {
    "esm2": "mamp_esm2",
    "glm2": "mamp_glm2",
    "esm2_mid": "mamp_esm2_mid",
    "alphafold_pair_reps": "mamp_alphafold_pair_reps",
    "esm2_with_receptor": "mamp_esm2_with_receptor",
    "glm2_with_receptor": "mamp_glm2_with_receptor",
    "esm2_with_receptor_single_seq": "mamp_esm2_with_receptor_single_seq",
    "glm2_with_receptor_single_seq": "mamp_glm2_with_receptor_single_seq",
    "amplify": "mamp_amplify",
    "amplify_with_receptor": "mamp_amplify_with_receptor",
    "esm2_with_receptor_attn_film": "mamp_esm2_with_receptor_attn_film",
    "esm2_contrast": "mamp_esm2_contrast"
}


def main(args):
    """
    Main training/evaluation function for the MAMP model.
    
    This function handles the complete training/evaluation pipeline including:
    1. Initialization of distributed training if enabled
    2. WandB logging setup
    3. Model and dataset preparation
    4. Training loop with periodic evaluation
    5. Final evaluation and metrics logging
    6. Optional k-fold cross-validation
    
    Args:
        args: Parsed command line arguments containing all configuration parameters
        
    The function operates in two main modes:
    1. Training Mode: Trains the model from scratch or from a checkpoint
    2. Evaluation Mode: Evaluates a pretrained model on test data
    
    The training process includes:
    - Regular model checkpointing
    - Periodic evaluation on test data
    - Logging of metrics and visualizations to WandB
    - Generation of ROC and PR curves
    """
    # Initialize distributed training
    misc.init_distributed_mode(args)
    
    # Setup WandB logging if enabled
    if not args.disable_wandb and misc.is_main_process():
        current_datetime = datetime.datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
        if args.eval_only_data_path:
            run_name = f"{wandb_dict[args.model]}-{Path(args.eval_only_data_path).stem}-{current_datetime}"
            tags = [args.model, str(Path(args.eval_only_data_path).stem), "eval"]
        else:
            run_name = f"{wandb_dict[args.model]}-{Path(args.data_dir).name}-{current_datetime}"
            tags = [args.model, str(Path(args.data_dir).name), "train"]
        wandb.init(
            project="mamp_ml",
            entity="dmstev-uc-berkeley",
            name=run_name,
            #resume="must",
            #entity=args.wandb_group,
            config=args,
            dir=args.output_dir,
            tags=tags
        )
    print(args)
    
    # Set device for training
    device = torch.device(args.device)

    # Set random seeds for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Initialize model and load checkpoint if specified
    model = model_dict[args.model](args)
    if args.model_checkpoint_path:
        state_dict = torch.load(args.model_checkpoint_path)["model"]
        model.load_state_dict(state_dict)
        
    # Initialize dataset class
    dataset = dataset_dict[args.model]
    if issubclass(dataset, AlphaFoldDataset):
        dataset = partial(dataset, name_to_x=torch.load(args.name_to_x))
        
    # Print model statistics
    n_params = sum(p.numel() for p in model.parameters())
    n_params_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    if args.eval_only_data_path:
        print(f"Evaluating a model with {n_params_grad:,} trainable parameters out of {n_params:,} parameters")
    else:
        print(f"Training {n_params_grad:,} of {n_params:,} parameters")

    # Get model's collate function for data loading
    collate_fn = model.collate_fn
    
    # Setup distributed training if enabled
    model.to(args.device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()

    # Initialize optimizer
    param_groups = misc.param_groups_weight_decay(model, args.weight_decay)
    optimizer = optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    # Load model state if resuming training
    misc.load_model(args, model_without_ddp, optimizer, None)

    # Prepare test dataset and dataloader
    if args.eval_only_data_path:
        eval_data_path = args.eval_only_data_path
    else:
        #eval_data_path = f"{args.data_dir}/test_data_with_bulkiness.csv"
        eval_data_path = f"{args.data_dir}/test_stratify.csv"
    test_df = pd.read_csv(eval_data_path)
    ds_test = dataset(df=test_df)
    print(f"{len(ds_test)=}")
    
    # Setup test data sampler
    if args.distributed and args.dist_eval:
        raise NotImplementedError
        sampler_test = torch.utils.data.DistributedSampler(
            ds_test, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler_test = torch.utils.data.SequentialSampler(ds_test)
        
    # Create test dataloader
    dl_test = torch.utils.data.DataLoader(
        ds_test,
        sampler=sampler_test,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
    )
    
    # If in evaluation-only mode, evaluate and exit
    if args.eval_only_data_path:
        metrics = {}
        metrics.update(evaluate(model, dl_test, device, args, args.output_dir))
        print(metrics)
        if not args.disable_wandb and misc.is_main_process():
            wandb.finish()
        exit()

    # Prepare training dataset and dataloader
    train_df = pd.read_csv(f"{args.data_dir}/train_stratify.csv")
    #train_df = pd.read_csv(f"{args.data_dir}/train_data_with_bulkiness.csv")
    ds_train = dataset(df=train_df)
    print(f"{len(ds_train)=}")
    
    # Setup training data sampler
    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            ds_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(ds_train)
    # print(f'{len(ds_train)=} {sampler_train.total_size=}')

    # Create training dataloader
    dl_train = torch.utils.data.DataLoader(
        ds_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
    )

    print(f"Start training for {args.epochs} epochs, saving to {args.output_dir}")
    start_time = time.time()
    
    # Initial evaluation before training
    args.current_epoch = 0
    evaluate(model, dl_test, device, args, args.output_dir)
    
    # Training loop
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            dl_train.sampler.set_epoch(epoch)
        train_one_epoch(model, dl_train, optimizer, device, epoch, args)
        
        # Update current epoch for plotting
        args.current_epoch = epoch + 1
        
        # Periodic evaluation
        if epoch % args.eval_period == args.eval_period - 1:
            evaluate(model, dl_test, device, args, args.output_dir)
            
        # Periodic checkpointing
        if epoch % args.save_period == args.save_period - 1:
            ckpt_path = misc.save_model(
                args, epoch, model, model_without_ddp, optimizer, None
            )
            print(f"Saved checkpoint to {ckpt_path}")
    
    # Print total training time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

    # Final evaluation
    args.current_epoch = "final"
    metrics = evaluate(model, dl_test, device, args, args.output_dir)

    if misc.is_main_process():
        print("Final metrics:", metrics)

    # disable wandb logging
    if not args.disable_wandb and misc.is_main_process():
        wandb.finish()

    # Optional k-fold cross-validation
    if args.cross_eval_kfold:
        if not args.disable_wandb and misc.is_main_process():
            run_name = f"{wandb_dict[args.model]}-{Path(args.data_dir).name}"
            wandb.init(
                project="mamp_ml",
                entity="dmstev-uc-berkeley",
                name=f"{run_name}_{args.cross_eval_kfold}cv",
                #group=args.wandb_group,
                config=args,
                dir=args.output_dir,
            )
        skf = StratifiedKFold(
            n_splits=args.cross_eval_kfold, random_state=42, shuffle=True
        )
        for i, train_idx, test_idx in enumerate(
            skf.split(ds_train.df, ds_train.df["ec3"])
        ):
            model = model_dict[args.model](args)
            model.to(args.device)
            model_without_ddp = model
            if args.distributed:
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[args.gpu], find_unused_parameters=True
                )
                model_without_ddp = model.module
                num_tasks = misc.get_world_size()
                global_rank = misc.get_rank()

            # Initialize optimizer for this fold
            param_groups = misc.param_groups_weight_decay(model, args.weight_decay)
            optimizer = optim.AdamW(
                param_groups, lr=args.lr, weight_decay=args.weight_decay
            )
            misc.load_model(args, model_without_ddp, optimizer, None)


            cv_ds_train = SeqAffDataset(df=ds_train.df.iloc[train_idx])
            ds_train.df.iloc[train_idx].to_csv(f"{out_dir}/train_stratify.csv", index=False)
            #ds_train.df.iloc[train_idx].to_csv(f"{out_dir}/train_data_with_bulkiness.csv", index=False)

            if args.distributed:
                cv_sampler_train = torch.utils.data.DistributedSampler(
                    cv_ds_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
                )
                print("Sampler_train = %s" % str(sampler_train))
            else:
                cv_sampler_train = torch.utils.data.RandomSampler(ds_train)

            # Create training dataloader
            cv_dl_train = torch.utils.data.DataLoader(
                cv_ds_train,
                sampler=cv_sampler_train,
                batch_size=args.batch_size,
                collate_fn=collate_fn,
            )
            cv_ds_test = SeqAffDataset(df=ds_train.df.iloc[test_idx])
            ds_train.df.iloc[test_idx].to_csv(f"{out_dir}/test_stratify.csv", index=False)
            #ds_train.df.iloc[test_idx].to_csv(f"{out_dir}/test_data_with_bulkiness.csv", index=False)
            cv_sampler_test = torch.utils.data.SequentialSampler(cv_ds_test)
            
            # Create test dataloader
            cv_dl_test = torch.utils.data.DataLoader(
                ds_test,
                sampler=cv_sampler_test,
                batch_size=args.batch_size,
                collate_fn=collate_fn,
            )
            
            # Training loop for this fold
            for epoch in range(args.start_epoch, args.epochs):
                if args.distributed:
                    cv_dl_train.sampler.set_epoch(epoch)
                train_one_epoch(model, cv_dl_train, optimizer, device, epoch, args)
                
                # Periodic evaluation
                if epoch % args.eval_period == args.eval_period - 1:
                    evaluate(model, cv_dl_test, device, args, args.output_dir)
                
                # Periodic checkpointing
                if epoch % args.save_period == args.save_period - 1:
                    ckpt_path = misc.save_model(
                        args, epoch, model, model_without_ddp, optimizer, None
                    )
                    print(f"Saved checkpoint to {ckpt_path}")
            print(train_idx, test_idx)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.eval_only_data_path:
        out_dir = Path(f"../eval_model_results/{args.model}{Path(args.eval_only_data_path).stem}")
    else:
        out_dir = Path(f"../model_results/{args.model}_{Path(args.data_dir).name}")
    out_dir.mkdir(exist_ok=True, parents=True)
    args.output_dir = out_dir
    main(args)

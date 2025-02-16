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

from models.esm_model import ESMModel
from models.esm_mid_model import ESMMidModel
from models.glm_model import GLMModel
from models.alphafold_model import AlphaFoldModel
from models.esm_with_receptor_model import ESMWithReceptorModel
from models.glm_with_receptor_model import GLMWithReceptorModel
from models.esm_with_receptor_single_seq_model import ESMWithReceptorSingleSeqModel
from models.glm_with_receptor_single_seq_model import GLMWithReceptorSingleSeqModel
from models.amp_with_receptor_model import AMPWithReceptorModel
from models.esm_with_receptor_attn_film_model import ESMWithReceptorAttnFilmModel
from models.amp_model import AMPModel
from models.esm_contrast_model import ESMContrastiveModel
from engine_train import train_one_epoch, evaluate
from datasets.seq_dataset import PeptideSeqDataset
from datasets.alphafold_dataset import AlphaFoldDataset
from datasets.seq_with_receptor_dataset import PeptideSeqWithReceptorDataset
import misc
from sklearn.model_selection import StratifiedKFold



def get_args_parser():
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
    parser.add_argument(
        "--backbone",
        default="esm2_t33_650M_UR50D",
        help="af|esm2_t33_650M_UR50D|esm_msa1b_t12_100M_UR50S",
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

    # data params
    parser.add_argument("--max_context_length", type=int, default=2000)
    parser.add_argument("--num_workers", default=10, type=int)

    # train params
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-9)
    parser.add_argument("--weight_decay", type=float, default=0.5)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--cross_eval_kfold", type=int)

    # loss params
    parser.add_argument("--lambda_single", type=float, default=0.1)
    parser.add_argument(
        "--loss_single_aug", type=str, default="none", help="none|tp|forrev"
    )
    parser.add_argument("--lambda_double", type=float, default=1.0)
    parser.add_argument("--double_subsample_destabilizing_ratio", type=float, default=8)
    parser.add_argument("--lambda_pos", type=float, default=4)

    # eval params
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--dist_eval", action="store_true")
    parser.add_argument("--eval_reverse", action="store_true")
    parser.add_argument(
        "--test", action="store_true", help="use data_path NOT eval_data_paths"
    )

    # resume params
    parser.add_argument("--finetune", default="", type=str)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--start_epoch", type=int, default=0)

    # logging params
    parser.add_argument("--save_pred_dict", action="store_true")
    parser.add_argument("--eval_period", type=int, default=10)
    parser.add_argument("--save_period", type=int, default=1000)
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--wandb_group", default="krasileva", type=str)
    parser.add_argument("--model_checkpoint_path", type=str)

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


model_dict = {
    "esm2": ESMModel,
    "glm2": GLMModel,
    "esm2_mid": ESMMidModel,
    "alphafold_pair_reps": AlphaFoldModel,
    "esm2_with_receptor": ESMWithReceptorModel,
    "glm2_with_receptor": GLMWithReceptorModel,
    "esm2_with_receptor_single_seq": ESMWithReceptorSingleSeqModel,
    "glm2_with_receptor_single_seq": GLMWithReceptorSingleSeqModel,
    "amplify": AMPModel,
    "amplify_with_receptor": AMPWithReceptorModel,
    "esm2_with_receptor_attn_film": ESMWithReceptorAttnFilmModel,
    "esm2_contrast": ESMContrastiveModel
}

dataset_dict = {
    "esm2": PeptideSeqDataset,
    "glm2": PeptideSeqDataset,
    "esm2_mid": PeptideSeqDataset,
    "alphafold_pair_reps": AlphaFoldDataset,
    "esm2_with_receptor": PeptideSeqWithReceptorDataset,
    "glm2_with_receptor": PeptideSeqWithReceptorDataset,
    "esm2_with_receptor_single_seq": PeptideSeqWithReceptorDataset,
    "glm2_with_receptor_single_seq": PeptideSeqWithReceptorDataset,
    "amplify": PeptideSeqDataset,
    "amplify_with_receptor": PeptideSeqWithReceptorDataset,
    "esm2_with_receptor_attn_film": PeptideSeqWithReceptorDataset,
    "esm2_contrast": PeptideSeqDataset
}

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
    misc.init_distributed_mode(args)
    if not args.disable_wandb and misc.is_main_process():
        current_datetime = datetime.datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
        if args.eval_only_data_path:
            run_name = f"{wandb_dict[args.model]}-{Path(args.eval_only_data_path).stem}-{current_datetime}"
            tags = [args.model, str(Path(args.eval_only_data_path).stem), "eval"]
        else:
            run_name = f"{wandb_dict[args.model]}-{Path(args.data_dir).name}-{current_datetime}"
            tags = [args.model, str(Path(args.data_dir).name), "train"]
        wandb.init(
            project="mamp",
            name=run_name,
            entity=args.wandb_group,
            config=args,
            dir=args.output_dir,
            tags = tags
        )
    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    ## prepare model
    model = model_dict[args.model](args)
    if args.model_checkpoint_path:
        state_dict = torch.load(args.model_checkpoint_path)["model"]
        model.load_state_dict(state_dict)
    dataset = dataset_dict[args.model]
    if issubclass(dataset, AlphaFoldDataset):
        dataset = partial(dataset, name_to_x=torch.load(args.name_to_x))
    n_params = sum(p.numel() for p in model.parameters())
    n_params_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    if args.eval_only_data_path:
        print(f"Evaluating a model with {n_params_grad:,} trainable parameters out of {n_params:,} parameters")
    else:
        print(f"Training {n_params_grad:,} of {n_params:,} parameters")

    # tokenizer = model.get_tokenizer()
    collate_fn = model.collate_fn
    ## prepare DDP
    model.to(args.device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()

    ## prepare optimizer
    param_groups = misc.param_groups_weight_decay(model, args.weight_decay)
    optimizer = optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    misc.load_model(args, model_without_ddp, optimizer, None)

    ## prepare test data
    if args.eval_only_data_path:
        eval_data_path = args.eval_only_data_path
    else:
        eval_data_path = f"{args.data_dir}/test.csv"
    test_df = pd.read_csv(eval_data_path)
    ds_test = dataset(df=test_df)
    print(f"{len(ds_test)=}")
    if args.distributed and args.dist_eval:
        raise NotImplementedError
        sampler_test = torch.utils.data.DistributedSampler(
            ds_test, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler_test = torch.utils.data.SequentialSampler(ds_test)
    dl_test = torch.utils.data.DataLoader(
        ds_test,
        sampler=sampler_test,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
    )
    if args.eval_only_data_path:
        metrics = {}
        metrics.update(evaluate(model, dl_test, device, args, args.output_dir))
        # if misc.is_main_process():
        #     metrics["copypasta"] = ",,".join(
        #         [metrics[f"{dl.dataset.name}_copypasta"] for dl in [dl_test]]
        #     )
        print(metrics)
        if not args.disable_wandb and misc.is_main_process():
            wandb.finish()
        exit()

    ## prepare train data
    train_df = pd.read_csv(f"{args.data_dir}/train.csv")
    ds_train = dataset(df=train_df)
    print(f"{len(ds_train)=}")
    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            ds_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(ds_train)
    # print(f'{len(ds_train)=} {sampler_train.total_size=}')

    # collate_fn = partial(protein_collate_fn, tokenizer=tokenizer)
    dl_train = torch.utils.data.DataLoader(
        ds_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
    )

    print(f"Start training for {args.epochs} epochs, saving to {args.output_dir}")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            dl_train.sampler.set_epoch(epoch)
        train_one_epoch(model, dl_train, optimizer, device, epoch, args)
        if epoch % args.eval_period == args.eval_period - 1:
            evaluate(model, dl_test, device, args, args.output_dir)
        if epoch % args.save_period == args.save_period - 1:
            ckpt_path = misc.save_model(
                args, epoch, model, model_without_ddp, optimizer, None
            )
            print(f"Saved checkpoint to {ckpt_path}")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

    metrics = {}
    metrics.update(evaluate(model, dl_test, device, args, args.output_dir))

    if misc.is_main_process():
        # metrics["copypasta"] = ",,".join(
        #     [metrics[f"{dl.dataset.name}_copypasta"] for dl in [dl_test]]
        # )
        print(metrics)

    if not args.disable_wandb and misc.is_main_process():
        # wandb.log({"copypasta": metrics["copypasta"]})
        wandb.finish()

    if args.cross_eval_kfold:
        if not args.disable_wandb and misc.is_main_process():
            run_name = f"{wandb_dict[args.model]}-{Path(args.data_dir).name}"
            wandb.init(
                project="mamp",
                name=f"{run_name}_{args.cross_eval_kfold}cv",
                group=args.wandb_group,
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

            param_groups = misc.param_groups_weight_decay(model, args.weight_decay)
            optimizer = optim.AdamW(
                param_groups, lr=args.lr, weight_decay=args.weight_decay
            )
            misc.load_model(args, model_without_ddp, optimizer, None)


            cv_ds_train = SeqAffDataset(df=ds_train.df.iloc[train_idx])
            ds_train.df.iloc[train_idx].to_csv(f"{out_dir}/train.csv", index=False)

            if args.distributed:
                cv_sampler_train = torch.utils.data.DistributedSampler(
                    cv_ds_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
                )
                print("Sampler_train = %s" % str(sampler_train))
            else:
                cv_sampler_train = torch.utils.data.RandomSampler(ds_train)

            cv_dl_train = torch.utils.data.DataLoader(
                cv_ds_train,
                sampler=cv_sampler_train,
                batch_size=args.batch_size,
                collate_fn=collate_fn,
            )
            cv_ds_test = SeqAffDataset(df=ds_train.df.iloc[test_idx])
            ds_train.df.iloc[test_idx].to_csv(f"{out_dir}/test.csv", index=False)
            cv_sampler_test = torch.utils.data.SequentialSampler(cv_ds_test)
            cv_dl_test = torch.utils.data.DataLoader(
                ds_test,
                sampler=cv_sampler_test,
                batch_size=args.batch_size,
                collate_fn=collate_fn,
            )
            for epoch in range(args.start_epoch, args.epochs):
                if args.distributed:
                    cv_dl_train.sampler.set_epoch(epoch)
                train_one_epoch(model, cv_dl_train, optimizer, device, epoch, args)
                if epoch % args.eval_period == args.eval_period - 1:
                    evaluate(model, cv_dl_test, device, args, args.output_dir)
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

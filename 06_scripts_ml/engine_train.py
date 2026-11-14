#-----------------------------------------------------------------------------------------------
# Krasileva Lab - Plant & Microbial Biology Department UC Berkeley
# Author: Danielle M. Stevens
# Last Updated: 07/06/2020
# Script Purpose: 
# Inputs: 
# Outputs: 
#-----------------------------------------------------------------------------------------------

"""
Training and evaluation engine for a deep learning model.
This module provides the core functionality for training and evaluating machine learning models,
particularly focused on multi-class classification tasks. It includes functionality for
loss computation, metric tracking, and visualization of results.
"""

######################################################################
# import packages and libraries
######################################################################

import math
import sys
import os
import numpy as np
import pandas as pd
from functools import partial
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.cross_entropy import CrossEntropyLoss
from losses.supcon import SupConLoss

from transformers.tokenization_utils_base import BatchEncoding
import misc
import wandb
from pathlib import Path
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
    f1_score,
    average_precision_score,
    top_k_accuracy_score,
    roc_curve,
    precision_recall_curve,
)
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator

# Match 08_model_analysis theme_clean_thin (Arial, thin lines, compact)
_CLASS_NAMES = ("Immunogenic", "Non-immunogenic", "Weakly immunogenic")
_CLASS_COLORS = ("#000000", "#8B0000", "#00008B")  # black / dark red / dark blue
_PLOT_LINEWIDTH = 0.9
_AXIS_LINEWIDTH = 0.35
_TICK_SIZE = 5.5
_LABEL_SIZE = 6.5
_LEGEND_SIZE = 4.5
# ~25% smaller square panel than prior (1.75); keep legend band height similar
_FIGSIZE = (1.31, 1.72)


def _apply_clean_thin_style(ax, *, full_box: bool = False, dense_ticks: bool = False):
    """Journal-style axes: thin spines, short ticks, no grid.

    Args:
        full_box: if True, draw all four spines; else L-frame only.
        dense_ticks: if True, major every 0.2 and minor every 0.1 on both axes.
    """
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True if full_box else side in ("bottom", "left"))
        if ax.spines[side].get_visible():
            ax.spines[side].set_linewidth(_AXIS_LINEWIDTH)
            ax.spines[side].set_color("black")

    if dense_ticks:
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.tick_params(
            axis="both",
            which="major",
            length=2.0,
            width=0.3,
            labelsize=_TICK_SIZE,
            colors="black",
            pad=1.5,
            direction="out",
        )
        ax.tick_params(
            axis="both",
            which="minor",
            length=1.0,
            width=0.25,
            colors="black",
            direction="out",
        )
    else:
        ax.tick_params(
            axis="both",
            which="major",
            length=1.5,
            width=0.3,
            labelsize=_TICK_SIZE,
            colors="black",
            pad=1.5,
        )

    ax.set_facecolor("white")
    ax.grid(False)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname("Arial")
        label.set_color("black")


def _style_legend(leg):
    leg.get_frame().set_linewidth(0)
    leg.get_frame().set_facecolor("none")
    for text in leg.get_texts():
        text.set_fontname("Arial")
        text.set_fontsize(_LEGEND_SIZE)
        text.set_color("black")


def _finalize_square_curve(fig, ax, xlabel: str, ylabel: str, out_path: Path):
    """Square axes (full box, dense ticks) with legend below; fonts unchanged."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_box_aspect(1)
    ax.set_xlabel(
        xlabel,
        fontsize=_LABEL_SIZE,
        fontname="Arial",
        color="black",
        labelpad=3,
    )
    ax.set_ylabel(
        ylabel,
        fontsize=_LABEL_SIZE,
        fontname="Arial",
        color="black",
        labelpad=2,
    )
    _apply_clean_thin_style(ax, full_box=True, dense_ticks=True)

    handles, labels = ax.get_legend_handles_labels()
    # Figure-level legend below the axis title with a clear gap (fonts unchanged)
    leg = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.06),
        ncol=2,
        fontsize=_LEGEND_SIZE,
        frameon=False,
        handlelength=0.9,
        handletextpad=0.3,
        columnspacing=0.7,
        labelspacing=0.2,
        borderaxespad=0.0,
    )
    _style_legend(leg)

    fig.subplots_adjust(left=0.22, right=0.98, top=0.98, bottom=0.30)
    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor="white",
    )
    plt.close(fig)


def move_to_device(obj, device):
    """
    Recursively moves all PyTorch tensors in a nested dictionary to the specified device.
    
    Args:
        obj: The object to move (can be a dict, list, tuple, tensor, or BatchEncoding)
        device: The PyTorch device to move the tensors to

    Returns:
        The same object structure with all tensors moved to the specified device
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, BatchEncoding):
        return BatchEncoding({k: move_to_device(v, device) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    else:
        return obj

# Dictionary mapping loss function names to their implementations
loss_dict = {
    "ce": CrossEntropyLoss(),     # Standard cross-entropy loss
    "supcon": SupConLoss()        # Supervised contrastive loss
}

def train_one_epoch(
    model: torch.nn.Module,
    dl,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args=None,
):
    """
    Trains the model for one epoch.
    
    Args:
        model: The neural network model to train
        dl: DataLoader containing the training data
        optimizer: The optimizer for updating model parameters
        device: The device (CPU/GPU) to use for training
        epoch: Current epoch number
        args: Additional arguments for training configuration
    
    Returns:
        dict: Dictionary containing averaged training metrics for the epoch
    """
    # Set model to training mode and reset optimizer
    model.train()  # Enables training-specific behaviors like dropout and batch norm
    optimizer.zero_grad()

    # Initialize metric logging
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = 10

    # Lists to store predictions and ground truth for the entire epoch
    lists = {"gt": [], "pr": [], "x": []}
    
    # Training loop over batches
    for batch_idx, batch in enumerate(metric_logger.log_every(dl, print_freq, header)):
        # Update learning rate according to schedule
        misc.adjust_learning_rate(optimizer, batch_idx / len(dl) + epoch, args)

        # Move batch to appropriate device
        batch = move_to_device(batch, device)
        
        # Forward pass
        output = model(batch['x'])
        all_losses = {}
        model_with_losses = model.module if hasattr(model, "module") else model
        
        # Calculate all specified losses
        for loss_name in model_with_losses.losses:
            losses = loss_dict[loss_name](output, batch)
            all_losses.update(losses)

        total_loss = sum(all_losses.values())

        # Check for invalid loss values
        if not math.isfinite(total_loss.item()):
            print("Loss is {}, stopping training".format(total_loss.item()))
            sys.exit(1)

        # Backward pass and optimization - Training-specific step
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # Gradient clipping
        optimizer.step()
        optimizer.zero_grad()

        # Get predictions and calculate metrics
        gt = batch['y']
        model_with_losses = model.module if hasattr(model, "module") else model
        preds = model_with_losses.get_pr(output)

        # Calculate statistics and update loggers
        stats = model_with_losses.get_stats(gt, preds, train=True)  # Training-specific metrics
        lr = optimizer.param_groups[0]["lr"]
        losses_detach = {f"train_{k}": v.cpu().item() for k, v in all_losses.items()}
        
        # Update metric logger
        metric_logger.update(lr=lr)
        metric_logger.update(loss=total_loss.item())
        metric_logger.update(**losses_detach)
        metric_logger.update(**stats)
        
        # Log to wandb if enabled
        if not args.disable_wandb and misc.is_main_process():
            wandb.log(
                {
                    "train_loss": total_loss.item(),
                    "lr": lr,
                    **losses_detach,
                    **stats,
                }
            )
            
        # Store batch results
        lists["gt"].append(batch['y'].cpu())
        lists["pr"].append(preds.cpu())
        model_with_losses = model.module if hasattr(model, "module") else model
        lists["x"].extend(model_with_losses.batch_decode(batch))

    # Concatenate all predictions and ground truth
    gt_all = torch.cat(lists["gt"])
    prob_all = torch.cat(lists["pr"])

    # Save epoch predictions
    torch.save(
        {
            "gt": gt_all,       
            "pr": prob_all,
            "x": lists["x"]
        },
        args.output_dir / "train_preds.pth",
    )
    
    # Synchronize metrics across processes and return averaged stats
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


######################################################################
# evaluate the model on a validation/test dataset
######################################################################

@torch.no_grad()  # Disables gradient computation for evaluation
def evaluate(model, dl, device, args, output_dir):
    """
    Evaluates the model on a validation/test dataset.
    
    Args:
        model: The neural network model to evaluate
        dl: DataLoader containing the evaluation data
        device: The device (CPU/GPU) to use for evaluation
        args: Additional arguments for evaluation configuration
        output_dir: Directory to save evaluation results and plots
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()  # Disables training-specific behaviors like dropout

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    # Lists to store predictions, ground truth, and losses
    lists = {"gt": [], "pr": [], "x": [], "loss": []}
    
    # Evaluation loop - no gradient computation or parameter updates
    for batch in metric_logger.log_every(dl, 10, header):
        batch = move_to_device(batch, device)
        output = model(batch['x'])
        
        # Calculate losses (for monitoring only, no backprop)
        all_losses = {}
        model_with_losses = model.module if hasattr(model, "module") else model
        for loss_name in model_with_losses.losses:
            losses = loss_dict[loss_name](output, batch)
            all_losses.update(losses)
            
        # Store individual losses
        for loss_name, loss_val in all_losses.items():
            if loss_name not in lists.keys():
                lists[loss_name] = []
            lists[loss_name].append(loss_val.cpu())
            
        # Get predictions
        preds = model_with_losses.get_pr(output)
        lists["gt"].append(batch['y'].cpu())
        lists["pr"].append(preds.cpu())
        lists["x"].extend(model_with_losses.batch_decode(batch))

        total_loss = sum(all_losses.values())
        lists['loss'].append(total_loss.cpu())

    # Process all predictions and calculate metrics
    gt_all = torch.cat(lists["gt"])
    prob_all = torch.cat(lists["pr"])
    mean_loss = float(np.mean(lists['loss']))

    model_with_losses = model.module if hasattr(model, "module") else model
    stats = model_with_losses.get_stats(gt_all, prob_all, train=False)  # Testing-specific metrics

    # Calculate average losses
    for loss_name, loss_val in all_losses.items():
        stats[f'test_{loss_name}'] = float(np.mean(lists[loss_name]))
    stats['test_loss'] = mean_loss

    # Create plots directory
    plots_dir = Path(output_dir) / 'plots'
    plots_dir.mkdir(exist_ok=True, parents=True)

    # ROC / PR curves — aesthetics aligned with theme_clean_thin
    # (08_model_analysis/00_visualize_input_data.R)
    epoch_tag = getattr(args, "current_epoch", "final")
    gt_onehot = np.eye(3)[gt_all.cpu().numpy()]
    pr_np = prob_all.cpu().numpy()

    rcParams["font.family"] = "Arial"
    rcParams["pdf.fonttype"] = 42  # editable text in Illustrator

    # --- ROC ---
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    for i in range(3):
        fpr, tpr, _ = roc_curve(gt_onehot[:, i], pr_np[:, i])
        try:
            class_auc = roc_auc_score(gt_onehot[:, i], pr_np[:, i])
        except ValueError:
            class_auc = float("nan")
        ax.plot(
            fpr,
            tpr,
            color=_CLASS_COLORS[i],
            linewidth=_PLOT_LINEWIDTH,
            solid_capstyle="butt",
            label=f"{_CLASS_NAMES[i]} ({class_auc:.2f})",
        )
    ax.plot(
        [0, 1],
        [0, 1],
        color="black",
        linestyle="--",
        linewidth=0.6,
        zorder=0,
    )
    _finalize_square_curve(
        fig,
        ax,
        xlabel="false positive rate",
        ylabel="true positive rate",
        out_path=plots_dir / f"roc_curve_epoch_{epoch_tag}.pdf",
    )

    # --- Precision-Recall ---
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    mean_recall = np.linspace(0, 1, 101)
    mean_precision = np.zeros_like(mean_recall)

    for i in range(3):
        precision, recall, _ = precision_recall_curve(gt_onehot[:, i], pr_np[:, i])
        ax.plot(
            recall,
            precision,
            color=_CLASS_COLORS[i],
            linewidth=_PLOT_LINEWIDTH,
            solid_capstyle="butt",
            label=f"{_CLASS_NAMES[i]} ({stats[f'test_auprc_class{i}']:.2f})",
        )
        mean_precision += np.interp(mean_recall, recall[::-1], precision[::-1])

    mean_precision /= 3
    mean_auprc = float(np.mean([stats[f"test_auprc_class{i}"] for i in range(3)]))
    ax.plot(
        mean_recall,
        mean_precision,
        color="black",
        linestyle="--",
        linewidth=0.6,
        label=f"mean ({mean_auprc:.2f})",
    )
    _finalize_square_curve(
        fig,
        ax,
        xlabel="recall",
        ylabel="precision",
        out_path=plots_dir / f"pr_curve_epoch_{epoch_tag}.pdf",
    )

    # Save evaluation metrics to CSV
    metrics = {
        'epoch': getattr(args, "current_epoch", "final"),
        'auroc': stats["test_auroc"],
#        'auprc_immunogenic': stats["test_auprc_class0"],
#        'auprc_non_immunogenic': stats["test_auprc_class1"],
#        'auprc_weakly_immunogenic': stats["test_auprc_class2"],
        'auprc_class0': stats["test_auprc_class0"],
        'auprc_class1': stats["test_auprc_class1"],
        'auprc_class2': stats["test_auprc_class2"],
        'accuracy': stats["test_acc"],
        'f1_macro': stats["test_f1_macro"],
        'f1_weighted': stats["test_f1_weighted"],
        'loss': stats["test_loss"]
    }
    
    # Load existing metrics if available, otherwise create new DataFrame
    metrics_file = plots_dir / 'test_metrics.csv'
    if metrics_file.exists():
        df_metrics = pd.read_csv(metrics_file)
        df_new = pd.DataFrame([metrics])
        df_metrics = pd.concat([df_metrics, df_new], ignore_index=True)
    else:
        df_metrics = pd.DataFrame([metrics])
    
    df_metrics.to_csv(metrics_file, index=False)

    # Create and save training progress visualization
    # Convert epoch column to numeric, replacing 'final' with the last numeric value + 1
    numeric_epochs = pd.to_numeric(
        df_metrics["epoch"].replace("final", float("inf")), errors="coerce"
    )
    if float("inf") in numeric_epochs.values:
        last_numeric = numeric_epochs[numeric_epochs != float("inf")].max()
        numeric_epochs = numeric_epochs.replace(
            float("inf"), last_numeric + 1 if not pd.isna(last_numeric) else 0
        )

    # Metric-progress panel — same clean_thin aesthetics as ROC/PR
    rcParams["font.family"] = "Arial"
    rcParams["pdf.fonttype"] = 42
    _metric_colors = ("#4A4A4A", "#8B0000", "#00008B")  # grey / red / blue for non-class series
    marker_kw = {"marker": "o", "markersize": 2.5, "linewidth": _PLOT_LINEWIDTH}

    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.2))

    # Plot 1: AUROC
    ax = axes[0, 0]
    ax.plot(numeric_epochs, df_metrics["auroc"], color="#000000", **marker_kw)
    ax.set_xlabel("epoch", fontsize=_LABEL_SIZE, fontname="Arial", color="black")
    ax.set_ylabel("auroc", fontsize=_LABEL_SIZE, fontname="Arial", color="black")
    _apply_clean_thin_style(ax)

    # Plot 2: AUPRC per class (same class colors as ROC/PR)
    ax = axes[0, 1]
    for i in range(3):
        ax.plot(
            numeric_epochs,
            df_metrics[f"auprc_class{i}"],
            color=_CLASS_COLORS[i],
            label=_CLASS_NAMES[i],
            **marker_kw,
        )
    ax.set_xlabel("epoch", fontsize=_LABEL_SIZE, fontname="Arial", color="black")
    ax.set_ylabel("auprc", fontsize=_LABEL_SIZE, fontname="Arial", color="black")
    _apply_clean_thin_style(ax)
    leg = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=2,
        fontsize=_LEGEND_SIZE,
        handlelength=1.2,
        borderpad=0.2,
    )
    _style_legend(leg)

    # Plot 3: Accuracy and F1
    ax = axes[1, 0]
    ax.plot(
        numeric_epochs,
        df_metrics["accuracy"],
        color=_metric_colors[0],
        label="accuracy",
        **marker_kw,
    )
    ax.plot(
        numeric_epochs,
        df_metrics["f1_macro"],
        color=_metric_colors[1],
        label="f1 macro",
        **marker_kw,
    )
    ax.plot(
        numeric_epochs,
        df_metrics["f1_weighted"],
        color=_metric_colors[2],
        label="f1 weighted",
        **marker_kw,
    )
    ax.set_xlabel("epoch", fontsize=_LABEL_SIZE, fontname="Arial", color="black")
    ax.set_ylabel("score", fontsize=_LABEL_SIZE, fontname="Arial", color="black")
    _apply_clean_thin_style(ax)
    leg = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=3,
        fontsize=_LEGEND_SIZE,
        handlelength=1.2,
        borderpad=0.2,
    )
    _style_legend(leg)

    # Plot 4: Loss
    ax = axes[1, 1]
    ax.plot(numeric_epochs, df_metrics["loss"], color="#000000", **marker_kw)
    ax.set_xlabel("epoch", fontsize=_LABEL_SIZE, fontname="Arial", color="black")
    ax.set_ylabel("loss", fontsize=_LABEL_SIZE, fontname="Arial", color="black")
    _apply_clean_thin_style(ax)

    fig.tight_layout(pad=0.6, h_pad=1.6, w_pad=0.8)
    fig.savefig(
        plots_dir / "test_progress.pdf",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    # Save predictions for later analysis
    torch.save(
        {
            "gt": gt_all,       # Ground truth labels
            "pr": prob_all,     # Model predictions
            "x": lists["x"]     # Input data
        },
        output_dir / "test_preds.pth",
    )

    # Log dataset name and metrics
    ds_name = dl.dataset.name if hasattr(dl.dataset, 'name') else 'test'
    print(ds_name, stats)

    # Update and return metrics
    metric_logger.update(**stats)
    ret = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return ret

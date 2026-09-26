#!/usr/bin/env python3
"""
MAMP-ML ESM attention maps (standalone — does not modify other scripts).

For each example, writes three PDFs under embedding_analysis_results/attention_maps/:

  1) *_receptor_only_attention.pdf
     Self-attention map for the receptor sequence alone.

  2) *_epitope_receptor_attention.pdf
     Peptide (epitope) → receptor attention from the joint peptide|EOS|receptor pass.

  3) *_attention_vs_bfactor.pdf
     Peptide→receptor attention profile overlaid with BFactorWeightGenerator
     weights and Filtered B-Factor from bfactor_winding_lrr_segments.csv.

Example:
  python 07_model_results/06_attention_maps.py \\
    --data_csv 05_datasets/train_data_with_all_train_immuno_stratify.csv \\
    --checkpoint 07_model_results/00_mamp_ml_best_params/checkpoint-19.pth \\
    --mamp_esm facebook/esm2_t6_8M_UR50D \\
    --bfactor_csv 04_Preprocessing_results/bfactor_winding_lrr_segments.csv \\
    --output_dir embedding_analysis_results \\
    --attn_layer 2 \\
    --max_examples 3 \\
    --device mps
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import BatchEncoding

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "06_scripts_ml"))

from models.esm_positon_weighted import (  # noqa: E402
    BFactorWeightGenerator,
    ESMBfactorWeightedFeatures,
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.linewidth": 0.35,
    "xtick.major.width": 0.3,
    "ytick.major.width": 0.3,
    "xtick.major.size": 1.5,
    "ytick.major.size": 1.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

CATEGORY_TO_INDEX = {
    "Immunogenic": 0,
    "Non-Immunogenic": 1,
    "Weakly Immunogenic": 2,
}
INDEX_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_INDEX.items()}
# None / 0 = full-resolution heatmaps (default). Set via --heatmap_max if needed.
DEFAULT_HEATMAP_MAX = None


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def parse_features(feature_str):
    if feature_str is None or (isinstance(feature_str, float) and pd.isna(feature_str)):
        return []
    if pd.isna(feature_str) or feature_str == "":
        return []
    text = str(feature_str).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []
    if text.startswith("c(") and text.endswith(")"):
        text = text[2:-1]
    values = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" in tok:
            tok = tok.split("=")[-1].strip()
        tok = tok.strip("()[] ")
        try:
            values.append(float(tok))
        except ValueError:
            continue
    return values


def row_to_example(row):
    outcome = row.get("Known Outcome", "Non-Immunogenic")
    y = CATEGORY_TO_INDEX.get(outcome, 1)
    receptor = str(row.get("Receptor", ""))
    receptor_plot = "INR" if receptor == "INR-like" else receptor
    epitope = str(row.get("Epitope", row.get("Ligand", "")))
    species = str(row.get("Plant species", row.get("Plant Species", "")))
    locus = str(row.get("Locus ID/Genbank", row.get("Locus ID", "")))
    receptor_id = f"{species}|{locus}|{receptor}" if locus else receptor
    return {
        "peptide_x": str(row["Sequence"]),
        "receptor_x": str(row["Receptor Sequence"]),
        "y": y,
        "receptor_id": receptor_id,
        "receptor": receptor_plot,
        "epitope": epitope,
        "known_outcome": INDEX_TO_CATEGORY.get(y, "Non-Immunogenic"),
        "sequence_bulkiness": parse_features(row.get("Sequence_Bulkiness", "")),
        "sequence_charge": parse_features(row.get("Sequence_Charge", "")),
        "sequence_hydrophobicity": parse_features(row.get("Sequence_Hydrophobicity", "")),
        "receptor_bulkiness": parse_features(row.get("Receptor_Bulkiness", "")),
        "receptor_charge": parse_features(row.get("Receptor_Charge", "")),
        "receptor_hydrophobicity": parse_features(row.get("Receptor_Hydrophobicity", "")),
    }


def filter_dataframe(df, args):
    out = df.copy()
    if args.receptor:
        out = out[out["Receptor"].astype(str).str.contains(args.receptor, case=False, na=False)]
    if args.epitope:
        epi_col = "Epitope" if "Epitope" in out.columns else "Ligand"
        out = out[out[epi_col].astype(str).str.contains(args.epitope, case=False, na=False)]
    if args.row_indices:
        idxs = [int(x) for x in args.row_indices.split(",") if x.strip()]
        out = df.iloc[idxs]
    out = out.reset_index(drop=True)
    if args.max_examples is not None:
        out = out.iloc[: args.max_examples].copy()
    return out.reset_index(drop=True)


def resolve_esm_name(name):
    name = str(name).strip()
    if "/" not in name:
        name = f"facebook/{name}"
    return name


def safe_slug(text, max_len=60):
    s = re.sub(r"[^A-Za-z0-9._+-]+", "_", str(text)).strip("_")
    return s[:max_len] or "example"


def short_esm_name(esm_name):
    name = str(esm_name)
    return name.split("/")[-1] if "/" in name else name


def format_model_line(esm_name, checkpoint, attn_layer=-1, n_layers=None, pretrained_only=False):
    if pretrained_only:
        ckpt = "pretrained ESM (zero-shot, no fine-tuning)"
    else:
        ckpt = Path(checkpoint).name if checkpoint else "checkpoint"
    if attn_layer is None or attn_layer == -1:
        layer_s = f"last (L{n_layers - 1})" if n_layers else "last"
    elif attn_layer < 0 and n_layers is not None:
        layer_s = f"{attn_layer} → absolute {n_layers + attn_layer}"
    else:
        layer_s = str(attn_layer)
    model_name = (
        "pretrained ESM (zero-shot)"
        if pretrained_only
        else "MAMP-ML (fine-tuned)"
    )
    return (
        f"Model: {model_name} · backbone {short_esm_name(esm_name)} · "
        f"{ckpt} · attn_layer={layer_s}"
    )


def layer_tag(attn_layer):
    """Filename-safe layer tag."""
    if attn_layer is None or attn_layer == -1:
        return "layer_last"
    if attn_layer < 0:
        return f"layer_m{-attn_layer}"  # e.g. -2 → layer_m2
    return f"layer_{attn_layer}"


def make_title(main, detail, example_line, model_line):
    return "\n".join([main, detail, example_line, model_line])


# ---------------------------------------------------------------------------
# Model / tensors
# ---------------------------------------------------------------------------
def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  warning: missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"  warning: unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    model.to(device)
    if device.type == "mps":
        for p in model.parameters():
            _ = p.numel()
        torch.mps.empty_cache()
    model.eval()
    return model


def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, BatchEncoding):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    return obj


def find_sep_index(input_ids, sep_id):
    ids = input_ids.detach().cpu().tolist()
    hits = [i for i, t in enumerate(ids) if t == sep_id]
    if not hits:
        return None
    return hits[0]


def peptide_receptor_slices(input_ids, mask, sep_id, drop_specials=True):
    m = mask.detach().cpu().numpy().astype(bool)
    sep = find_sep_index(input_ids, sep_id)
    if sep is None:
        raise RuntimeError("Separator (EOS) not found in joint sequence")
    all_idx = np.where(m)[0]
    pep = all_idx[all_idx < sep]
    rec = all_idx[all_idx > sep]
    if drop_specials:
        if len(pep) > 1:
            pep = pep[1:]  # drop <cls>
        if len(rec) > 1:
            rec = rec[:-1]  # drop trailing EOS
    return pep, rec, sep


def lookup_bfactor_arrays(bfactor_gen, protein_key, sequence_length):
    """
    Match BFactorWeightGenerator key resolution, then build:
      - weights: same as get_weights(...)
      - filtered: Filtered B-Factor placed at CSV residue indices (0 elsewhere)
    """
    weights = bfactor_gen.get_weights(protein_key, sequence_length).cpu().numpy()
    filtered = np.zeros(sequence_length, dtype=np.float64)

    data = None
    if protein_key in bfactor_gen.bfactor_data:
        data = bfactor_gen.bfactor_data[protein_key]
    else:
        converted = protein_key.replace("|", "_").replace(" ", "_")
        if converted in bfactor_gen.bfactor_data:
            data = bfactor_gen.bfactor_data[converted]

    if data is not None:
        for idx, bf in zip(data["residue_idx"], data["bfactors"]):
            idx = int(idx)
            if 0 <= idx < sequence_length:
                filtered[idx] = float(bf)
    return weights, filtered


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _style_ax(ax):
    ax.set_facecolor("white")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(0.35)
    ax.tick_params(
        axis="both", which="both", direction="out", length=1.5, width=0.3,
        colors="black", labelsize=6, top=False, right=False,
    )


def save_fig(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {path}")


def downsample_matrix(mat, max_n=None):
    """Return matrix (optionally downsampled) and original-axis indices for ticks."""
    mat = np.asarray(mat, dtype=np.float64)
    if mat.ndim != 2 or mat.size == 0:
        return mat, np.array([]), np.array([])
    nr, nc = mat.shape
    if max_n is None or max_n <= 0:
        row_idx = np.arange(nr)
        col_idx = np.arange(nc)
    else:
        row_idx = (
            np.arange(nr) if nr <= max_n else np.linspace(0, nr - 1, max_n).astype(int)
        )
        col_idx = (
            np.arange(nc) if nc <= max_n else np.linspace(0, nc - 1, max_n).astype(int)
        )
    return mat[np.ix_(row_idx, col_idx)], row_idx, col_idx


def _set_residue_ticks(ax, row_idx, col_idx, n_ticks=6):
    """
    Label heatmap axes with original residue/token indices (1-based display).
    """
    def _ticks(idx):
        if len(idx) == 0:
            return [], []
        if len(idx) <= n_ticks:
            pos = np.arange(len(idx))
        else:
            pos = np.linspace(0, len(idx) - 1, n_ticks).astype(int)
        labels = [str(int(idx[p]) + 1) for p in pos]
        return pos, labels

    xpos, xlab = _ticks(col_idx)
    ypos, ylab = _ticks(row_idx)
    if len(xpos):
        ax.set_xticks(xpos)
        ax.set_xticklabels(xlab, fontsize=6)
    if len(ypos):
        ax.set_yticks(ypos)
        ax.set_yticklabels(ylab, fontsize=6)


def plot_heatmap(
    mat,
    title,
    path,
    xlabel="Key residue",
    ylabel="Query residue",
    log_scale=True,
    heatmap_max=None,
):
    mat = np.asarray(mat, dtype=np.float64)
    if mat.size == 0 or not np.isfinite(mat).any() or np.nanmax(mat) <= 0:
        print(f"  skip heatmap (empty/zero): {path}")
        return

    plot_mat = np.log10(mat + 1e-8) if log_scale else mat
    mat_ds, row_idx, col_idx = downsample_matrix(plot_mat, max_n=heatmap_max)
    finite = mat_ds[np.isfinite(mat_ds)]
    vmax = float(np.percentile(finite, 99)) if finite.size else 1.0
    vmin = float(np.percentile(finite, 5)) if finite.size else 0.0
    if vmax <= vmin:
        vmax = float(finite.max()) if finite.size else 1.0
        vmin = float(finite.min()) if finite.size else 0.0

    nr, nc = mat_ds.shape
    n_orig_r, n_orig_c = mat.shape
    down_r = heatmap_max and n_orig_r > heatmap_max
    down_c = heatmap_max and n_orig_c > heatmap_max
    width = min(7.0, max(3.2, 0.008 * nc + 2.0))
    height = min(5.0, max(2.4, 0.06 * nr + 1.6))
    fig, ax = plt.subplots(figsize=(width, height), dpi=300)
    fig.patch.set_facecolor("white")
    im = ax.imshow(
        mat_ds,
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )
    x_note = f"{xlabel} (1…{n_orig_c}" + ("; downsampled)" if down_c else ")")
    y_note = f"{ylabel} (1…{n_orig_r}" + ("; downsampled)" if down_r else ")")
    ax.set_title(title, fontsize=7, color="black", pad=6, loc="left")
    ax.set_xlabel(x_note, fontsize=7)
    ax.set_ylabel(y_note, fontsize=7)
    _set_residue_ticks(ax, row_idx, col_idx)
    _style_ax(ax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log10(attn)" if log_scale else "attn", fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    fig.tight_layout(pad=0.5)
    save_fig(fig, path)


def plot_receptor_only_attention_figure(attn, title, path, heatmap_max=None):
    """
    Two-panel receptor-only view:
      top  — log10 self-attention heatmap
      bottom — mean incoming attention profile
    """
    attn = np.asarray(attn, dtype=np.float64)
    if attn.ndim != 2 or attn.shape[0] < 2:
        print(f"  skip receptor-only figure: {path}")
        return

    incoming = attn.mean(axis=0)
    n = attn.shape[0]
    offdiag = incoming.copy()
    for i in range(n):
        offdiag[i] = (attn[:, i].sum() - attn[i, i]) / max(n - 1, 1)

    fig, axes = plt.subplots(
        2, 1, figsize=(5.4, 4.6), dpi=300,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    fig.patch.set_facecolor("white")

    plot_mat = np.log10(attn + 1e-8)
    mat_ds, row_idx, col_idx = downsample_matrix(plot_mat, max_n=heatmap_max)
    finite = mat_ds[np.isfinite(mat_ds)]
    vmax = float(np.percentile(finite, 99))
    vmin = float(np.percentile(finite, 5))
    im = axes[0].imshow(
        mat_ds, aspect="auto", interpolation="nearest",
        cmap="magma", vmin=vmin, vmax=vmax,
    )
    axes[0].set_title(title, fontsize=7, color="black", pad=6, loc="left")
    down = heatmap_max and n > heatmap_max
    x_lab = f"Receptor key residue (1…{n}" + ("; downsampled)" if down else ")")
    y_lab = f"Receptor query residue (1…{n}" + ("; downsampled)" if down else ")")
    axes[0].set_xlabel(x_lab, fontsize=7)
    axes[0].set_ylabel(y_lab, fontsize=7)
    _set_residue_ticks(axes[0], row_idx, col_idx)
    _style_ax(axes[0])
    cbar = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    cbar.set_label("log10(attn)", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    xs = np.arange(1, n + 1)
    axes[1].plot(xs, incoming, color="#4c78a8", linewidth=0.8, label="Mean incoming attn")
    axes[1].plot(xs, offdiag, color="#e07a3d", linewidth=0.8, label="Mean off-diagonal incoming")
    axes[1].set_xlabel("Receptor residue index", fontsize=8)
    axes[1].set_ylabel("Attention", fontsize=8)
    _style_ax(axes[1])
    axes[1].legend(fontsize=5.5, frameon=False, loc="best")

    fig.tight_layout(pad=0.5)
    save_fig(fig, path)


def plot_attention_vs_bfactor(xs, attn, weights, filtered_bf, title, path):
    """Receptor-token attention profile vs B-factor weight and Filtered B-Factor."""
    fig, axes = plt.subplots(2, 1, figsize=(5.2, 3.4), dpi=300, sharex=True)
    fig.patch.set_facecolor("white")

    ax0, ax1 = axes
    ax0.plot(xs, attn, color="#4c78a8", linewidth=0.9, label="Peptide→receptor attn")
    ax0.set_ylabel("Attention", fontsize=8, color="#4c78a8")
    ax0.tick_params(axis="y", labelcolor="#4c78a8", labelsize=6)
    ax0.set_title(title, fontsize=7, color="black", pad=6, loc="left")
    _style_ax(ax0)
    ax0.legend(fontsize=5.5, frameon=False, loc="best")

    ax1.plot(
        xs, weights, color="#e07a3d", linewidth=0.9,
        label="BFactorWeightGenerator weight",
    )
    ax1.plot(
        xs, filtered_bf, color="#59a14f", linewidth=0.8, alpha=0.9,
        label="Filtered B-Factor",
    )
    ax1.set_xlabel("Receptor token index (joint sequence)", fontsize=8)
    ax1.set_ylabel("B-factor signals", fontsize=8)
    _style_ax(ax1)
    ax1.legend(fontsize=5.5, frameon=False, loc="best")

    fig.tight_layout(pad=0.5)
    save_fig(fig, path)


# ---------------------------------------------------------------------------
# Attention extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def esm_attentions(esm, input_ids, attention_mask, layer=-1):
    out = esm(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
        return_dict=True,
    )
    layer_idx = layer if layer >= 0 else len(out.attentions) + layer
    attn = out.attentions[layer_idx][0].detach().float().cpu().numpy()  # (H, L, L)
    mat = attn.mean(axis=0)
    mask = attention_mask[0].detach().cpu().numpy().astype(bool)
    mat = mat.copy()
    mat[~mask, :] = 0.0
    mat[:, ~mask] = 0.0
    return mat


def mean_peptide_to_receptor(attn, pep_idx, rec_idx):
    if len(pep_idx) == 0 or len(rec_idx) == 0:
        return np.zeros(len(rec_idx))
    return attn[np.ix_(pep_idx, rec_idx)].mean(axis=0)


def process_example(
    example,
    model,
    tokenizer,
    device,
    out_dir,
    example_tag,
    esm_name,
    checkpoint,
    bfactor_gen,
    attn_layer=-1,
    heatmap_max=None,
    pretrained_only=False,
):
    n_layers = len(model.esm.encoder.layer)
    # Resolve negative indices for display (e.g. -2 on 6-layer model → 4)
    abs_layer = attn_layer if attn_layer >= 0 else n_layers + attn_layer
    if abs_layer < 0 or abs_layer >= n_layers:
        raise ValueError(
            f"--attn_layer {attn_layer} out of range for this ESM "
            f"({n_layers} layers, valid 0..{n_layers - 1} or -1..-{n_layers})"
        )
    model_line = format_model_line(
        esm_name,
        checkpoint,
        attn_layer=attn_layer,
        n_layers=n_layers,
        pretrained_only=pretrained_only,
    )
    ltag = layer_tag(attn_layer)
    if pretrained_only:
        ltag = f"zeroshot_{ltag}"

    collated = model.collate_fn([example])
    batch = move_to_device(collated["x"], device)
    if pretrained_only:
        example_line = (
            f"Example: {example['receptor']} / {example['epitope']} | "
            f"true={example['known_outcome']} · pred=n/a (zero-shot ESM, no classifier)"
        )
        pred = -1
    else:
        logits = model(batch)
        pred = int(logits.argmax(dim=-1)[0].item())
        example_line = (
            f"Example: {example['receptor']} / {example['epitope']} | "
            f"true={example['known_outcome']} · MAMP-ML pred={INDEX_TO_CATEGORY.get(pred, pred)}"
        )
    print(f"  using attention layer {abs_layer}/{n_layers - 1} (--attn_layer {attn_layer})")

    # ------------------------------------------------------------------
    # Plot 1 — receptor-only attention map
    # ------------------------------------------------------------------
    enc_rec = tokenizer(
        example["receptor_x"],
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=1024,
    )
    enc_rec = move_to_device(enc_rec, device)
    attn_rec = esm_attentions(
        model.esm, enc_rec["input_ids"], enc_rec["attention_mask"], layer=attn_layer
    )
    mask_rec = enc_rec["attention_mask"][0].detach().cpu().numpy().astype(bool)
    keep = np.where(mask_rec)[0]
    content = keep[1:-1] if len(keep) > 2 else keep
    if len(content) > 1:
        rec_block = attn_rec[np.ix_(content, content)]
        print(
            f"  receptor-only attention: shape={rec_block.shape} "
            f"diag_mean={float(np.diag(rec_block).mean()):.4f} "
            f"max={float(rec_block.max()):.4f}"
        )
        plot_receptor_only_attention_figure(
            rec_block,
            make_title(
                "Receptor-only attention map",
                f"Self-attention on receptor alone (L={rec_block.shape[0]} residues); "
                "log heatmap + profiles (raw maps look similar because attn is local)",
                example_line,
                model_line,
            ),
            out_dir / f"{example_tag}_{ltag}_receptor_only_attention.pdf",
            heatmap_max=heatmap_max,
        )
    else:
        print("  warning: too few receptor tokens for receptor-only map")

    # ------------------------------------------------------------------
    # Plot 2 — epitope + receptor (joint) peptide→receptor attention map
    # ------------------------------------------------------------------
    input_ids = batch["combined_tokens"]
    attn_mask = batch["combined_mask"]
    pep_idx, rec_idx, sep = peptide_receptor_slices(
        input_ids[0], attn_mask[0], model.separator_token_id, drop_specials=True
    )
    if len(pep_idx) == 0 or len(rec_idx) == 0:
        print("  warning: empty peptide or receptor slice; skipping joint attention plots")
        return

    attn_joint = esm_attentions(model.esm, input_ids, attn_mask, layer=attn_layer)
    cross = attn_joint[np.ix_(pep_idx, rec_idx)]
    print(
        f"  epitope→receptor attention: shape={cross.shape} "
        f"mean={float(cross.mean()):.5f} max={float(cross.max()):.4f}"
    )
    plot_heatmap(
        cross,
        make_title(
            "Epitope + receptor attention map",
            "Peptide (epitope) → receptor cross-attention from joint peptide|EOS|receptor input "
            "(log10 scale)",
            example_line,
            model_line,
        ),
        out_dir / f"{example_tag}_{ltag}_epitope_receptor_attention.pdf",
        xlabel="Receptor residue",
        ylabel="Epitope residue",
        log_scale=True,
        heatmap_max=heatmap_max,
    )

    # ------------------------------------------------------------------
    # Plot 3 — peptide→receptor attention vs B-factor
    # ------------------------------------------------------------------
    pep_to_rec = mean_peptide_to_receptor(attn_joint, pep_idx, rec_idx)
    full_len = int(input_ids.shape[1])
    weights_full, filtered_full = lookup_bfactor_arrays(
        bfactor_gen, example["receptor_id"], full_len
    )
    weights_rec = weights_full[rec_idx]
    filtered_rec = filtered_full[rec_idx]
    xs = np.arange(len(rec_idx))

    def _norm(v):
        v = np.asarray(v, dtype=np.float64)
        span = np.nanmax(np.abs(v)) if v.size else 0.0
        return v / (span + 1e-12)

    plot_attention_vs_bfactor(
        xs,
        _norm(pep_to_rec),
        _norm(weights_rec),
        _norm(filtered_rec),
        make_title(
            "Receptor-token attention vs B-factor",
            "Peptide→receptor attn (norm) with BFactorWeightGenerator weights "
            "& Filtered B-Factor (norm)",
            example_line,
            model_line,
        ),
        out_dir / f"{example_tag}_{ltag}_attention_vs_bfactor.pdf",
    )

    np.savez_compressed(
        out_dir / f"{example_tag}_{ltag}_attention_arrays.npz",
        receptor_only_attention=attn_rec[np.ix_(content, content)] if len(content) > 1 else np.array([]),
        epitope_receptor_attention=cross,
        peptide_to_receptor_profile=pep_to_rec,
        peptide_indices=pep_idx,
        receptor_indices=rec_idx,
        sep_index=np.array([sep]),
        bfactor_weights_receptor=weights_rec,
        filtered_bfactor_receptor=filtered_rec,
        true_y=np.array([example["y"]]),
        pred_y=np.array([pred]),
        attn_layer=np.array([abs_layer]),
        n_layers=np.array([n_layers]),
        model_name=np.array(
            ["pretrained-ESM-zeroshot" if pretrained_only else "MAMP-ML"]
        ),
        esm_backbone=np.array([str(esm_name)]),
        checkpoint=np.array(
            ["none-pretrained" if pretrained_only else str(checkpoint)]
        ),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="MAMP-ML receptor-only and epitope+receptor attention maps"
    )
    p.add_argument(
        "--data_csv",
        default="05_datasets/train_data_with_all_train_immuno_stratify.csv",
    )
    p.add_argument(
        "--checkpoint",
        default="07_model_results/00_mamp_ml_best_params/checkpoint-19.pth",
    )
    p.add_argument("--mamp_esm", default="facebook/esm2_t6_8M_UR50D")
    p.add_argument(
        "--bfactor_csv",
        default="04_Preprocessing_results/bfactor_winding_lrr_segments.csv",
        help="B-factor CSV for BFactorWeightGenerator / Filtered B-Factor overlay",
    )
    p.add_argument(
        "--output_dir",
        default="embedding_analysis_results",
        help="Root output directory (PDFs go under attention_maps/)",
    )
    p.add_argument("--max_examples", type=int, default=3)
    p.add_argument("--receptor", default=None)
    p.add_argument("--epitope", default=None)
    p.add_argument("--row_indices", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--attn_layer",
        type=int,
        default=-1,
        help="ESM attention layer index for all attention plots. "
             "0 = first layer; -1 = last (default). "
             "esm2_t6_8M has layers 0..5.",
    )
    p.add_argument(
        "--heatmap_max",
        type=int,
        default=None,
        help="Optional max pixels per heatmap axis for display only "
             "(e.g. 180). Default: no downsampling (full residue resolution).",
    )
    p.add_argument(
        "--pretrained_only",
        action="store_true",
        help="Zero-shot: use the HuggingFace pretrained ESM backbone only "
             "(skip fine-tuned checkpoint). Attention from frozen ESM; "
             "no MAMP-ML classifier predictions.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    out_dir = Path(args.output_dir) / "attention_maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    esm_name = resolve_esm_name(args.mamp_esm)

    print(f"Loading data: {args.data_csv}")
    df = pd.read_csv(args.data_csv)
    df = filter_dataframe(df, args)
    print(f"  examples to plot: {len(df)}")
    if len(df) == 0:
        raise SystemExit("No examples after filtering.")

    # Build MAMP wrapper for tokenizer/collate; ESM starts as HuggingFace pretrained.
    # With --pretrained_only we skip loading the fine-tuned .pth so attentions
    # come from zero-shot ESM weights.
    print(f"Loading ESM backbone: {esm_name}")
    mamp_args = SimpleNamespace(bfactor_csv_path=args.bfactor_csv, esm_model=esm_name)
    model = ESMBfactorWeightedFeatures(args=mamp_args, num_classes=3)
    if args.pretrained_only:
        print("  mode: zero-shot pretrained ESM (not loading fine-tuned checkpoint)")
        model.to(device)
        model.eval()
        for p in model.esm.parameters():
            p.requires_grad = False
        checkpoint_label = "pretrained_zeroshot"
    else:
        print(f"  loading fine-tuned checkpoint: {args.checkpoint}")
        load_checkpoint(model, Path(args.checkpoint), device)
        checkpoint_label = args.checkpoint
    tokenizer = model.tokenizer

    print(f"Loading B-factor table: {args.bfactor_csv}")
    bfactor_gen = BFactorWeightGenerator(
        bfactor_csv_path=args.bfactor_csv, min_weight=0.5, max_weight=2.0
    )

    for i, (_, row) in enumerate(df.iterrows()):
        example = row_to_example(row)
        tag = f"{i:02d}_{safe_slug(example['receptor'])}_{safe_slug(example['epitope'])}"
        print(f"\n=== Example {i + 1}/{len(df)}: {tag} ===")
        process_example(
            example=example,
            model=model,
            tokenizer=tokenizer,
            device=device,
            out_dir=out_dir,
            example_tag=tag,
            esm_name=esm_name,
            checkpoint=checkpoint_label,
            bfactor_gen=bfactor_gen,
            attn_layer=args.attn_layer,
            heatmap_max=args.heatmap_max,
            pretrained_only=args.pretrained_only,
        )

    print(f"\nDone. Three PDFs per example in {out_dir}/")
    print("  *_receptor_only_attention.pdf")
    print("  *_epitope_receptor_attention.pdf")
    print("  *_attention_vs_bfactor.pdf")


if __name__ == "__main__":
    main()

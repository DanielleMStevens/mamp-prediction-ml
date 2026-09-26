#!/usr/bin/env python3
"""
Extract pre-classifier embeddings from SeqOnly (ESMWithReceptorModel),
MAMP-ML (ESMBfactorWeightedFeatures), and a pretrained-only ESM baseline
(no fine-tuning), then plot 2D projections colored by known outcome and
by receptor-epitope group.

Does NOT modify model source files — fine-tuned embeddings are captured with
forward hooks on the classification head. The pretrained baseline uses frozen
ESM CLS tokens only (peptide ∥ receptor).

Example:
  python 07_model_results/05_umap_embeddings.py \
    --train_csv 05_datasets/train_data_with_all_train_immuno_stratify.csv \
    --seqonly_checkpoint ../model_results/02_immuno_stratify_esm2_with_receptor/checkpoint-14.pth \
    --seqonly_esm facebook/esm2_t30_150M_UR50D \
    --mamp_checkpoint 07_model_results/00_mamp_ml_best_params/checkpoint-19.pth \
    --mamp_esm facebook/esm2_t6_8M_UR50D \
    --bfactor_csv 04_Preprocessing_results/bfactor_winding_lrr_segments.csv \
    --output_dir embedding_analysis_results \
    --method pca \
    --reuse_embeddings \
    --device mps
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# Limit BLAS/OpenMP threads before numpy/sklearn load — prevents OpenBLAS
# segfaults that sometimes hit sklearn TSNE on macOS/arm64.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, BatchEncoding

# ---------------------------------------------------------------------------
# Paths / imports
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "06_scripts_ml"))

from models.esm_with_receptor_model import ESMWithReceptorModel  # noqa: E402
from models.esm_positon_weighted import ESMBfactorWeightedFeatures  # noqa: E402

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    from openTSNE import TSNE as OpenTSNE
    HAS_OPENTSNE = True
except ImportError:
    HAS_OPENTSNE = False

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Matplotlib journal style (matches theme_clean_thin in R figures)
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

# Project palette (aligned with other figures)
OUTCOME_COLORS = {
    "Immunogenic": "#e07a3d",
    "Non-Immunogenic": "#4c78a8",
    "Weakly Immunogenic": "#59a14f",
}
RECEPTOR_COLORS = {
    "CuRe1": "#b35c46",
    "CORE": "#e2b048",
    "EFR": "#ebd56d",
    "FLS2": "#b9d090",
    "FLS3": "#37a170",
    "INR": "#86c0ce",
    "RLP23": "#7d9fc6",
    "PERU": "#32527B",
    "RLP42": "#542a64",
    "MIK2": "#232232",
    "NUT": "#D5869D",
}
CATEGORY_TO_INDEX = {
    "Immunogenic": 0,
    "Non-Immunogenic": 1,
    "Weakly Immunogenic": 2,
}
INDEX_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_INDEX.items()}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TrainEmbeddingDataset:
    """Minimal dataset matching model collate_fn expectations + metadata."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        def parse_features(feature_str):
            """Parse comma-separated floats; skip malformed tokens (e.g. R leftovers)."""
            if feature_str is None or (isinstance(feature_str, float) and pd.isna(feature_str)):
                return []
            if pd.isna(feature_str) or feature_str == "":
                return []
            text = str(feature_str).strip()
            if not text or text.lower() in {"nan", "none"}:
                return []
            # Strip occasional R-style wrappers like c(...)
            if text.startswith("c(") and text.endswith(")"):
                text = text[2:-1]
            values = []
            for tok in text.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                # Drop junk like "P = 17.43" → keep trailing number if present
                if "=" in tok:
                    tok = tok.split("=")[-1].strip()
                tok = tok.strip("()[] ")
                try:
                    values.append(float(tok))
                except ValueError:
                    continue
            return values

        outcome = row.get("Known Outcome", "Non-Immunogenic")
        y = CATEGORY_TO_INDEX.get(outcome, 1)
        receptor = str(row.get("Receptor", ""))
        if receptor == "INR-like":
            receptor_plot = "INR"
        else:
            receptor_plot = receptor
        epitope = str(row.get("Epitope", row.get("Ligand", "")))
        species = str(row.get("Plant species", row.get("Plant Species", "")))
        locus = str(row.get("Locus ID/Genbank", row.get("Locus ID", "")))
        # Match BFactorWeightGenerator training key when possible
        receptor_id = f"{species}|{locus}|{receptor}" if locus else receptor

        return {
            "peptide_x": row["Sequence"],
            "receptor_x": row["Receptor Sequence"],
            "y": y,
            "receptor_id": receptor_id,
            "receptor": receptor_plot,
            "epitope": epitope,
            "receptor_epitope": f"{receptor_plot} / {epitope}",
            "known_outcome": INDEX_TO_CATEGORY.get(y, "Non-Immunogenic"),
            "sequence_bulkiness": parse_features(row.get("Sequence_Bulkiness", "")),
            "sequence_charge": parse_features(row.get("Sequence_Charge", "")),
            "sequence_hydrophobicity": parse_features(
                row.get("Sequence_Hydrophobicity", "")
            ),
            "receptor_bulkiness": parse_features(row.get("Receptor_Bulkiness", "")),
            "receptor_charge": parse_features(row.get("Receptor_Charge", "")),
            "receptor_hydrophobicity": parse_features(
                row.get("Receptor_Hydrophobicity", "")
            ),
        }


def collate_with_meta(model_collate, batch):
    """Run model collate and keep metadata lists aligned with the batch."""
    meta = {
        "receptor": [b["receptor"] for b in batch],
        "epitope": [b["epitope"] for b in batch],
        "receptor_epitope": [b["receptor_epitope"] for b in batch],
        "known_outcome": [b["known_outcome"] for b in batch],
        "y": [b["y"] for b in batch],
    }
    collated = model_collate(batch)
    return collated, meta


# ---------------------------------------------------------------------------
# Model loading / embedding extraction (hooks only — no model edits)
# ---------------------------------------------------------------------------
def load_checkpoint(model, path, device):
    # Always load weights on CPU first, then move the full model to device.
    # Loading directly onto MPS can leave unallocated placeholder storage.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    # Strip DDP prefix if present
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  warning: missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"  warning: unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    # Catch ESM-size mismatches early (hidden dim / layer count differ)
    size_mismatches = [
        k for k in missing + unexpected
        if any(tok in k for tok in ("esm", "net.", "classifier.", "film"))
    ]
    if len(size_mismatches) > 10:
        raise RuntimeError(
            f"Checkpoint {path} does not match the constructed model "
            f"(likely wrong --seqonly_esm / --mamp_esm). "
            f"Mismatched keys include: {size_mismatches[:8]}..."
        )
    model.to(device)
    # Force materialize parameters/buffers on MPS (avoids placeholder storage errors)
    if device.type == "mps":
        for p in model.parameters():
            _ = p.numel()
        for b in model.buffers():
            _ = b.numel()
        torch.mps.empty_cache()
    model.eval()
    return model


def move_batch_to_device(batch, device):
    """Recursively move tensors; convert BatchEncoding to a plain dict on device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, BatchEncoding):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [move_batch_to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(v, device) for v in batch)
    return batch


def extract_embeddings_with_hook(model, dataloader, device, head_attr):
    """
    Capture inputs to the classification head via a forward hook.
    head_attr: 'net' for SeqOnly, 'classifier' for MAMP-ML.
    """
    head = getattr(model, head_attr)
    captured = []

    def _hook(_module, inputs, _output):
        # inputs[0] is the pre-classifier embedding tensor
        captured.append(inputs[0].detach().cpu())

    handle = head.register_forward_hook(_hook)

    all_embeddings = []
    meta_rows = []

    with torch.no_grad():
        for batch_idx, (batch, meta) in enumerate(dataloader):
            batch = move_batch_to_device(batch, device)
            captured.clear()
            _ = model(batch["x"])
            if not captured:
                raise RuntimeError(
                    f"No embeddings captured from '{head_attr}' on batch {batch_idx}"
                )
            emb = captured[0]
            all_embeddings.append(emb.numpy())
            n = emb.shape[0]
            for i in range(n):
                meta_rows.append(
                    {
                        "receptor": meta["receptor"][i],
                        "epitope": meta["epitope"][i],
                        "receptor_epitope": meta["receptor_epitope"][i],
                        "known_outcome": meta["known_outcome"][i],
                        "y": meta["y"][i],
                    }
                )
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                print(f"  batch {batch_idx + 1}/{len(dataloader)} "
                      f"(n={len(meta_rows)})")

    handle.remove()
    embeddings = np.vstack(all_embeddings)
    meta_df = pd.DataFrame(meta_rows)
    return embeddings, meta_df


def extract_pretrained_esm_embeddings(
    dataset,
    device,
    batch_size=8,
    model_name="facebook/esm2_t30_150M_UR50D",
):
    """
    Frozen pretrained ESM only — no fine-tuned FiLM / classifier.

    Pair representation = concat(peptide CLS, receptor CLS).
    Uses the same backbone as SeqOnly by default (esm2_t30_150M).
    """
    print(f"Loading pretrained ESM (no fine-tuning): {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    esm = AutoModel.from_pretrained(model_name)
    esm.to(device)
    esm.eval()
    for p in esm.parameters():
        p.requires_grad = False

    def _collate(batch):
        meta = {
            "receptor": [b["receptor"] for b in batch],
            "epitope": [b["epitope"] for b in batch],
            "receptor_epitope": [b["receptor_epitope"] for b in batch],
            "known_outcome": [b["known_outcome"] for b in batch],
            "y": [b["y"] for b in batch],
        }
        peptide = tokenizer(
            [b["peptide_x"] for b in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        receptor = tokenizer(
            [b["receptor_x"] for b in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return {"peptide_x": peptide, "receptor_x": receptor}, meta

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
    )

    all_embeddings = []
    meta_rows = []
    with torch.no_grad():
        for batch_idx, (batch, meta) in enumerate(loader):
            batch = move_batch_to_device(batch, device)
            pep = esm(**batch["peptide_x"]).last_hidden_state[:, 0, :]
            rec = esm(**batch["receptor_x"]).last_hidden_state[:, 0, :]
            # Concatenate CLS tokens — no learned fusion
            emb = torch.cat([pep, rec], dim=-1).detach().cpu().numpy()
            all_embeddings.append(emb)
            n = emb.shape[0]
            for i in range(n):
                meta_rows.append(
                    {
                        "receptor": meta["receptor"][i],
                        "epitope": meta["epitope"][i],
                        "receptor_epitope": meta["receptor_epitope"][i],
                        "known_outcome": meta["known_outcome"][i],
                        "y": meta["y"][i],
                    }
                )
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                print(
                    f"  batch {batch_idx + 1}/{len(loader)} "
                    f"(n={len(meta_rows)})"
                )

    del esm
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    return np.vstack(all_embeddings), pd.DataFrame(meta_rows)


# ---------------------------------------------------------------------------
# Dimensionality reduction + plotting
# ---------------------------------------------------------------------------
def _pca_preprocess(x, seed=42):
    n_pca = min(50, x.shape[0] - 1, x.shape[1])
    if n_pca >= 2 and x.shape[1] > 2:
        print(f"PCA preprocess to {n_pca} dims...")
        return PCA(n_components=n_pca, random_state=seed).fit_transform(x)
    return x


def _sanitize_for_tsne(x, seed=42):
    """
    Prepare features for t-SNE to avoid a known sklearn Barnes–Hut segfault:
    identical / near-identical points + PCA init can produce NaNs in C++ code.
    """
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        print("  replacing non-finite embedding values with 0")
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # Drop zero-variance columns (can break PCA init / BH tree)
    std = x.std(axis=0)
    keep = std > 1e-12
    if keep.sum() < x.shape[1]:
        dropped = int((~keep).sum())
        print(f"  dropping {dropped} near-constant feature(s)")
        x = x[:, keep] if keep.any() else x

    # Tiny jitter breaks exact duplicates (same workaround used by openTSNE)
    rng = np.random.RandomState(seed)
    scale = float(np.std(x)) or 1.0
    x = x + (1e-6 * scale) * rng.standard_normal(size=x.shape)
    return np.ascontiguousarray(x, dtype=np.float64)


def _run_tsne(x, seed=42):
    """Run t-SNE with crash-resistant settings; prefer openTSNE when available."""
    x = _sanitize_for_tsne(x, seed=seed)
    perplexity = min(30, max(5, (len(x) - 1) // 3))

    # Cap BLAS threads again in case imports already started pools
    try:
        from threadpoolctl import threadpool_limits
        thread_ctx = threadpool_limits(limits=1)
    except Exception:
        thread_ctx = None

    def _sklearn_tsne():
        # init="random" avoids PCA-init NaN path; n_jobs=1 avoids OpenMP crashes
        return TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=seed,
            init="random",
            learning_rate="auto",
            method="barnes_hut",
            angle=0.5,
            max_iter=1000,
            n_jobs=1,
            verbose=1,
        ).fit_transform(x)

    try:
        if thread_ctx is not None:
            thread_ctx.__enter__()
        if HAS_OPENTSNE:
            print(
                f"Running t-SNE via openTSNE "
                f"(n={len(x)}, perplexity={perplexity})..."
            )
            coords = np.asarray(
                OpenTSNE(
                    n_components=2,
                    perplexity=perplexity,
                    random_state=seed,
                    n_jobs=1,
                    verbose=True,
                ).fit(x)
            )
        else:
            print(
                f"Running t-SNE via sklearn "
                f"(n={len(x)}, perplexity={perplexity}, init=random)..."
            )
            print(
                "  tip: conda install -c conda-forge opentsne "
                "for a more robust t-SNE backend"
            )
            coords = _sklearn_tsne()
    finally:
        if thread_ctx is not None:
            thread_ctx.__exit__(None, None, None)

    return coords


def reduce_2d(embeddings, method="umap", seed=42):
    """Reduce embeddings to 2D. method: 'pca' | 'umap' | 'tsne'."""
    method = method.lower().replace("-", "").replace(" ", "")
    if method not in {"pca", "umap", "tsne"}:
        raise ValueError(f"--method must be pca, umap, or tsne; got {method!r}")

    x = np.asarray(embeddings, dtype=np.float64)

    if method == "pca":
        print("Running PCA (2D)...")
        coords = PCA(n_components=2, random_state=seed).fit_transform(x)
        return coords, "PCA"

    x_pca = _pca_preprocess(x, seed=seed)

    if method == "umap":
        if not HAS_UMAP:
            raise ImportError(
                "UMAP requested but umap-learn is not installed. "
                "Install with: conda install -c conda-forge umap-learn"
            )
        print("Running UMAP...")
        reducer = umap.UMAP(
            n_neighbors=min(15, max(2, len(x_pca) - 1)),
            min_dist=0.1,
            n_components=2,
            metric="cosine",
            random_state=seed,
            low_memory=True,
        )
        coords = reducer.fit_transform(x_pca)
        return coords, "UMAP"

    coords = _run_tsne(x_pca, seed=seed)
    return coords, "t-SNE"


def _style_ax(ax):
    """Thin L-frame, white background, no grid (matches theme_clean_thin)."""
    ax.set_facecolor("white")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(0.35)
    ax.tick_params(
        axis="both",
        which="both",
        direction="out",
        length=1.5,
        width=0.3,
        colors="black",
        labelsize=7,
        top=False,
        right=False,
    )


def _scatter(ax, x, y, labels, color_map, title, method="UMAP", legend_ncol=1, point_size=6):
    unique = list(dict.fromkeys(labels))
    for lab in unique:
        mask = np.array([l == lab for l in labels])
        color = color_map.get(lab, None)
        ax.scatter(
            x[mask],
            y[mask],
            s=point_size,
            alpha=0.8,
            label=lab,
            c=[color] if color is not None else None,
            edgecolors="none",
            linewidths=0,
        )
    ax.set_title(title, fontsize=8, color="black", pad=4)
    ax.set_xlabel(f"{method} 1", fontsize=8, color="black")
    ax.set_ylabel(f"{method} 2", fontsize=8, color="black")
    _style_ax(ax)
    leg = ax.legend(
        fontsize=5.5,
        frameon=False,
        loc="best",
        ncol=legend_ncol,
        markerscale=1.2,
        handletextpad=0.3,
        borderaxespad=0.2,
        labelspacing=0.25,
    )
    if leg is not None:
        for text in leg.get_texts():
            text.set_color("black")


def _receptor_color_map(receptors):
    receptor_map = {**RECEPTOR_COLORS}
    leftover = sorted(set(receptors) - set(receptor_map))
    if leftover:
        cmap = plt.get_cmap("tab20")
        for i, r in enumerate(leftover):
            receptor_map[r] = cmap(i % 20)
    return receptor_map


def plot_model_umaps(coords, meta_df, model_name, method, output_path, n_examples):
    """Two-panel figure: outcome | receptor group."""
    x, y = coords[:, 0], coords[:, 1]
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.5), dpi=300)
    fig.patch.set_facecolor("white")

    _scatter(
        axes[0], x, y, meta_df["known_outcome"].tolist(), OUTCOME_COLORS,
        "Known Outcome", method=method, legend_ncol=1,
    )
    receptor_map = _receptor_color_map(meta_df["receptor"])
    _scatter(
        axes[1], x, y, meta_df["receptor"].tolist(), receptor_map,
        "Receptor group", method=method, legend_ncol=2,
    )

    fig.suptitle(
        f"{model_name} ({method}, n={n_examples})",
        fontsize=9, color="black", y=1.02,
    )
    fig.tight_layout(pad=0.4)
    fig.savefig(output_path, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_comparison(model_results, method, output_path):
    """
    Rows = models, cols = outcome | receptor.
    model_results: list of (name, coords, meta_df)
    """
    n = len(model_results)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 2, figsize=(5.6, 2.6 * n), dpi=300)
    if n == 1:
        axes = np.array([axes])
    fig.patch.set_facecolor("white")

    all_receptors = []
    for _, _, meta in model_results:
        all_receptors.extend(meta["receptor"].tolist())
    receptor_map = _receptor_color_map(all_receptors)

    for row, (name, coords, meta) in enumerate(model_results):
        _scatter(
            axes[row, 0], coords[:, 0], coords[:, 1],
            meta["known_outcome"].tolist(), OUTCOME_COLORS,
            f"{name} - Known Outcome", method=method, legend_ncol=1,
        )
        _scatter(
            axes[row, 1], coords[:, 0], coords[:, 1],
            meta["receptor"].tolist(), receptor_map,
            f"{name} - Receptor group", method=method, legend_ncol=2,
        )

    names = " vs ".join(name for name, _, _ in model_results)
    fig.suptitle(f"{names} ({method})", fontsize=9, color="black", y=1.01)
    fig.tight_layout(pad=0.4)
    fig.savefig(output_path, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="2D projection (PCA/UMAP/t-SNE) of SeqOnly vs MAMP-ML training embeddings"
    )
    p.add_argument(
        "--train_csv",
        default="05_datasets/train_data_with_all_train_immuno_stratify.csv",
        help="Training CSV with sequences, labels, and chemical features",
    )
    p.add_argument(
        "--seqonly_checkpoint",
        default="../model_results/02_immuno_stratify_esm2_with_receptor/checkpoint-14.pth",
        help="Checkpoint for SeqOnly (ESMWithReceptorModel)",
    )
    p.add_argument(
        "--mamp_checkpoint",
        default="07_model_results/00_mamp_ml_best_params/checkpoint-19.pth",
        help="Checkpoint for MAMP-ML (ESMBfactorWeightedFeatures)",
    )
    p.add_argument(
        "--bfactor_csv",
        default="04_Preprocessing_results/bfactor_winding_lrr_segments.csv",
        help="B-factor CSV required by MAMP-ML model",
    )
    p.add_argument(
        "--output_dir",
        default="embedding_analysis_results",
        help="Directory for embeddings CSVs and UMAP plots",
    )
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap for a quick smoke test",
    )
    p.add_argument(
        "--skip_seqonly",
        action="store_true",
        help="Only run MAMP-ML (useful if SeqOnly checkpoint is unavailable)",
    )
    p.add_argument(
        "--skip_mamp",
        action="store_true",
        help="Only run SeqOnly",
    )
    p.add_argument(
        "--skip_pretrained",
        action="store_true",
        help="Skip pretrained-only ESM baseline (no fine-tuning)",
    )
    p.add_argument(
        "--seqonly_esm",
        default="facebook/esm2_t30_150M_UR50D",
        help="ESM backbone for SeqOnly; must match the checkpoint "
             "(default: facebook/esm2_t30_150M_UR50D)",
    )
    p.add_argument(
        "--mamp_esm",
        default="facebook/esm2_t6_8M_UR50D",
        help="ESM backbone for MAMP-ML; must match the checkpoint "
             "(default: facebook/esm2_t6_8M_UR50D)",
    )
    p.add_argument(
        "--pretrained_esm",
        default="facebook/esm2_t30_150M_UR50D",
        help="HuggingFace ESM id for the no-fine-tuning baseline "
             "(default: same backbone as SeqOnly)",
    )
    p.add_argument(
        "--method",
        choices=["pca", "umap", "tsne"],
        default="pca",
        help="2D projection method: pca, umap, or tsne (default: pca)",
    )
    p.add_argument(
        "--reuse_embeddings",
        action="store_true",
        help="Skip extraction if {Model}_train_embeddings.npy already exists; "
             "only run 2D projection + plotting",
    )
    return p.parse_args()


def project_and_plot(model_name, embeddings, meta_df, output_dir, method="pca"):
    """2D projection + PDF plots from already-extracted embeddings."""
    coords, method_label = reduce_2d(embeddings, method=method)
    tag = method_label.lower().replace("-", "")
    coord_df = meta_df.copy()
    coord_df[f"{method_label}1"] = coords[:, 0]
    coord_df[f"{method_label}2"] = coords[:, 1]
    # Keep UMAP1/UMAP2 aliases for downstream convenience
    coord_df["UMAP1"] = coords[:, 0]
    coord_df["UMAP2"] = coords[:, 1]
    coord_df.to_csv(
        output_dir / f"{model_name}_{tag}_coordinates.csv", index=False
    )
    plot_model_umaps(
        coords,
        meta_df,
        model_name,
        method_label,
        output_dir / f"{model_name}_{tag}.pdf",
        n_examples=len(meta_df),
    )
    return coords, meta_df, method_label


def run_one_model(
    model_name,
    model,
    head_attr,
    dataset,
    device,
    batch_size,
    output_dir,
):
    print(f"\n=== Extracting embeddings: {model_name} ===")

    def _collate(batch):
        return collate_with_meta(model.collate_fn, batch)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
    )
    embeddings, meta_df = extract_embeddings_with_hook(
        model, loader, device, head_attr=head_attr
    )
    print(f"  embeddings shape: {embeddings.shape}")

    emb_path = output_dir / f"{model_name}_train_embeddings.npy"
    meta_path = output_dir / f"{model_name}_train_metadata.csv"
    np.save(emb_path, embeddings)
    meta_df.to_csv(meta_path, index=False)
    print(f"  wrote {emb_path}")
    print(f"  wrote {meta_path}")
    return embeddings, meta_df


def load_saved_embeddings(model_name, output_dir):
    emb_path = output_dir / f"{model_name}_train_embeddings.npy"
    meta_path = output_dir / f"{model_name}_train_metadata.csv"
    if not emb_path.exists() or not meta_path.exists():
        return None
    print(f"Reusing saved embeddings: {emb_path}")
    embeddings = np.load(emb_path)
    meta_df = pd.read_csv(meta_path)
    return embeddings, meta_df


def main():
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"Loading training data: {args.train_csv}")
    df = pd.read_csv(args.train_csv)
    if args.max_samples is not None:
        df = df.iloc[: args.max_samples].copy()
        print(f"  using first {len(df)} rows (--max_samples)")
    else:
        print(f"  n = {len(df)}")

    dataset = TrainEmbeddingDataset(df)
    results = {}

    # Preferred plot order: pretrained baseline → SeqOnly → MAMP-ML
    if not args.skip_pretrained:
        saved = (
            load_saved_embeddings("ESM-pretrained", output_dir)
            if args.reuse_embeddings
            else None
        )
        if saved is not None:
            embeddings, meta_df = saved
        else:
            print("\n=== Extracting embeddings: ESM-pretrained (no fine-tuning) ===")
            embeddings, meta_df = extract_pretrained_esm_embeddings(
                dataset,
                device,
                batch_size=args.batch_size,
                model_name=args.pretrained_esm,
            )
            print(f"  embeddings shape: {embeddings.shape}")
            emb_path = output_dir / "ESM-pretrained_train_embeddings.npy"
            meta_path = output_dir / "ESM-pretrained_train_metadata.csv"
            np.save(emb_path, embeddings)
            meta_df.to_csv(meta_path, index=False)
            print(f"  wrote {emb_path}")
            print(f"  wrote {meta_path}")

        results["ESM-pretrained"] = project_and_plot(
            "ESM-pretrained", embeddings, meta_df, output_dir, method=args.method
        )

    if not args.skip_seqonly:
        saved = (
            load_saved_embeddings("SeqOnly", output_dir)
            if args.reuse_embeddings
            else None
        )
        if saved is not None:
            embeddings, meta_df = saved
        else:
            seq_path = Path(args.seqonly_checkpoint)
            if not seq_path.exists():
                raise FileNotFoundError(f"SeqOnly checkpoint not found: {seq_path}")
            print(f"Loading SeqOnly from {seq_path}")
            print(f"  ESM backbone: {args.seqonly_esm}")
            seq_model = ESMWithReceptorModel(
                args=SimpleNamespace(esm_model=args.seqonly_esm)
            )
            load_checkpoint(seq_model, seq_path, device)
            embeddings, meta_df = run_one_model(
                "SeqOnly",
                seq_model,
                "net",
                dataset,
                device,
                args.batch_size,
                output_dir,
            )
            del seq_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()

        results["SeqOnly"] = project_and_plot(
            "SeqOnly", embeddings, meta_df, output_dir, method=args.method
        )

    if not args.skip_mamp:
        saved = (
            load_saved_embeddings("MAMP-ML", output_dir)
            if args.reuse_embeddings
            else None
        )
        if saved is not None:
            embeddings, meta_df = saved
        else:
            mamp_path = Path(args.mamp_checkpoint)
            if not mamp_path.exists():
                raise FileNotFoundError(f"MAMP-ML checkpoint not found: {mamp_path}")
            print(f"Loading MAMP-ML from {mamp_path}")
            print(f"  ESM backbone: {args.mamp_esm}")
            mamp_args = SimpleNamespace(
                bfactor_csv_path=args.bfactor_csv,
                esm_model=args.mamp_esm,
            )
            mamp_model = ESMBfactorWeightedFeatures(args=mamp_args, num_classes=3)
            load_checkpoint(mamp_model, mamp_path, device)
            embeddings, meta_df = run_one_model(
                "MAMP-ML",
                mamp_model,
                "classifier",
                dataset,
                device,
                args.batch_size,
                output_dir,
            )
            del mamp_model
            if device.type == "mps":
                torch.mps.empty_cache()

        results["MAMP-ML"] = project_and_plot(
            "MAMP-ML", embeddings, meta_df, output_dir, method=args.method
        )

    # Multi-model comparison in a fixed order when ≥2 models ran
    order = ["ESM-pretrained", "SeqOnly", "MAMP-ML"]
    present = [(name, *results[name]) for name in order if name in results]
    if len(present) >= 2:
        method = present[0][3]
        tag = method.lower().replace("-", "")
        # Full comparison of whatever ran
        plot_comparison(
            [(name, coords, meta) for name, coords, meta, _ in present],
            method,
            output_dir / f"embedding_comparison_{tag}.pdf",
        )
        # Keep legacy SeqOnly vs MAMP-ML filename when both present
        if "SeqOnly" in results and "MAMP-ML" in results:
            plot_comparison(
                [
                    ("SeqOnly", results["SeqOnly"][0], results["SeqOnly"][1]),
                    ("MAMP-ML", results["MAMP-ML"][0], results["MAMP-ML"][1]),
                ],
                method,
                output_dir / f"SeqOnly_vs_MAMP-ML_{tag}.pdf",
            )

    print("\nDone.")


if __name__ == "__main__":
    main()

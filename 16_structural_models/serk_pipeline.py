#!/usr/bin/env python3
"""
SERK co-receptor assignment pipeline.

Assigns a BAK1/SERK3-clade co-receptor to every LRR-PRR / ligand record,
preferring literature-validated genes over inference, and recording the
derivation of each row.

Tier 1   literature ground truth (native annotation identifiers)
Tier 2   ortholog from the receptor's own genome, anchored on tier-1 sequences
Tier 3a  transient expression host (heterologous assay systems)
Tier 3b  closest ground-truth substitution by taxonomic distance

Usage (from repo root or from this directory):
    python 16_structural_models/serk_pipeline.py

Defaults resolve relative to the repository root:
    --input   02_in_data/All_LRR_PRR_ligand_data.xlsx
    --workdir 16_structural_models/serk_work
    --out     16_structural_models/SERK_coreceptor.tsv

Network access is required. The full run downloads ~39 proteomes
(several GB of transfer, roughly 60-90 min). Intermediate JSON and
downloaded proteomes are cached in --workdir so the run is resumable.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd
import parasail
from Bio import SeqIO

# Repo root = parent of 16_structural_models/
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_INPUT = REPO_ROOT / "02_in_data" / "All_LRR_PRR_ligand_data.xlsx"
DEFAULT_WORKDIR = SCRIPT_DIR / "serk_work"
DEFAULT_OUT = SCRIPT_DIR / "SERK_coreceptor.tsv"

UA = {"User-Agent": "serk-coreceptor-pipeline"}
ENSEMBL_FTP = "https://ftp.ebi.ac.uk/ensemblgenomes/pub/plants/current/fasta/"
ENSEMBL_REST = "https://rest.ensembl.org"
UNIPROT = "https://rest.uniprot.org"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


def log(msg: str = "") -> None:
    """Print a status statement to the terminal (flushed immediately)."""
    print(msg, flush=True)


# --------------------------------------------------------------------------
# 1. Alignment primitive
# --------------------------------------------------------------------------
# Global Needleman-Wunsch, BLOSUM62, gap open 11 / extend 1.
# Percent identity = matches / aligned columns, excluding columns gapped in both.

def pid(a: str, b: str) -> float:
    tr = parasail.nw_trace_scan_16(a, b, 11, 1, parasail.blosum62).traceback
    matches = sum(1 for x, y in zip(tr.query, tr.ref) if x == y and x != "-")
    cols = sum(1 for x, y in zip(tr.query, tr.ref) if not (x == "-" and y == "-"))
    return 100.0 * matches / cols if cols else 0.0


# --------------------------------------------------------------------------
# 2. Reference proteins
# --------------------------------------------------------------------------
# AtSERK5 is a Col-0 pseudogene: used for clade verification only, never as a
# query. SOBIR1 and NIK1 are outgroup controls. AtSERK1/AtSERK2 form the
# subtrahend of the clade margin -- they are NOT rejection controls, because
# the At SERK1/2 vs SERK3/4/5 boundary does not transfer outside Brassicaceae.

REFS = {
    "Q94AG2": "AtSERK1",
    "Q9XIC7": "AtSERK2",
    "Q94F62": "AtBAK1/SERK3",
    "Q9SKG5": "AtSERK4/BKK1",
    "Q8LPS5": "AtSERK5",
    "Q9SKB2": "AtSOBIR1",
    "Q9LFS4": "AtNIK1",
}
SERK12 = ["Q94AG2", "Q9XIC7"]
OUTGROUP = ["Q9SKB2", "Q9LFS4"]
BAK1_CLADE = ["Q94F62", "Q9SKG5", "Q8LPS5"]


def fetch_refs(workdir: str) -> dict[str, str]:
    path = os.path.join(workdir, "refs.json")
    if os.path.exists(path):
        log(f"Loading cached reference sequences from {path}")
        return json.load(open(path))
    log(f"Fetching {len(REFS)} reference sequences from UniProt ...")
    out = {}
    for i, acc in enumerate(REFS, 1):
        log(f"  [{i}/{len(REFS)}] Downloading {REFS[acc]} ({acc})")
        url = f"{UNIPROT}/uniprotkb/{acc}.fasta"
        txt = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read().decode()
        out[acc] = "".join(txt.split("\n")[1:])
        time.sleep(0.3)
    json.dump(out, open(path, "w"))
    log(f"Saved reference sequences to {path}")
    return out


# --------------------------------------------------------------------------
# 3. Tier 1 - literature ground truth
# --------------------------------------------------------------------------
# Each entry: the published gene, its identifier in its native annotation, the
# second BAK1-clade gene where one is published, and the citation. Sequences
# are pulled from the annotation the identifier belongs to (see fetch_tier1).

TIER1 = {
    "Arabidopsis thaliana": dict(
        gene="BAK1/SERK3", native="AT4G33430", alt="AT2G13790 (SERK4/BKK1)",
        source="Araport11 / Ensembl AT4G33430.1",
        ref="Chinchilla et al. 2007 Nature doi:10.1038/nature05999; "
            "Heese et al. 2007 PNAS doi:10.1073/pnas.0705306104",
        bridge="none - published ID already in the receptor namespace"),
    "Solanum lycopersicum": dict(
        gene="SlSERK3A", native="Solyc10g047140", alt="Solyc01g104970 (SlSERK3B)",
        source="ITAG4.0_proteins.fasta",
        ref="Li et al. 2024 New Phytologist (Solyc IDs); Peng & Kaloshian 2014 "
            "PLoS ONE doi:10.1371/journal.pone.0093302 (flg22-dependent co-IP with SlFLS2)",
        bridge="none - published ID already in the receptor namespace"),
    "Glycine max": dict(
        gene="GmBAK1a", native="Glyma.15G051600", alt="Glyma.08G180800 (GmBAK1b)",
        source="Ensembl Plants Glycine_max_v2.1",
        ref="Zhang et al. 2024 Phytopathology Research doi:10.1186/s42483-024-00232-1",
        bridge="none - published ID already in the receptor namespace"),
    "Nicotiana benthamiana": dict(
        gene="NbSerk3A", native="Niben101Scf11779g01002",
        alt="Niben101Scf00279g02022 (second homeolog)",
        source="Niben101_annotation.proteins",
        ref="Chaparro-Garcia et al. 2011 PLoS ONE doi:10.1371/journal.pone.0016608",
        bridge="BRIDGED: clones predate the Niben genome; cDNA-derived proteins aligned "
               "to Niben101 - NbSERK3A 95.4%, NbSERK3B 93.7% to the same model "
               "(next best Niben101Scf00279g02022 92.0%). A/B not resolvable, "
               "the two clones are 98.8% identical"),
    "Solanum tuberosum": dict(
        gene="StSERK3A (Sotub10g013940)", native="PGSC0003DMG400017690",
        alt="PGSC0003DMG400012594 (StSERK3B, Sotub01g042020)",
        source="Ensembl Plants SolTub_3.0",
        ref="Ziegler et al. 2019 Sci Rep doi:10.1038/s41598-019-54944-y",
        bridge="BRIDGED: paper reports Sotub IDs; linked via transcript "
               "PGSC0003DMT400032797 = Sotub01g042020 and by chromosome (10 / 1). "
               "NOTE both PGSC models are truncated (525/518 aa vs ~615 typical)"),
    "Oryza sativa": dict(
        gene="OsSERK2 (Os04g38480 MSU)", native="Os04g0457800",
        alt="Os08g0174700 (OsSERK1/OsBAK1, developmental paralog)",
        source="RAP-DB via Ensembl",
        ref="Chen et al. 2014 Mol Plant PMID 24482436 (required for XA21-, XA3- "
            "and OsFLS2-mediated immunity)",
        bridge="none - published ID already in the receptor namespace. Overrides the "
               "ortholog call, which returns OsSERK1/OsBAK1"),
}

# Species checked in the literature sweep with no usable identifier. Recorded
# because each is a trap for name-based mining.
LIT_NOTES = {
    "Medicago truncatula":
        "NEGATIVE: the published Medicago gene is MtSERK1 (GenBank AY162177), the "
        "AtSERK1 orthologue, NOT a BAK1/SERK3 gene.",
    "Vigna unguiculata":
        "NEGATIVE: no cowpea BAK1 characterised. INR work used heterologous "
        "NbBAK1/NbSOBIR1 (Steinbrenner 2020 PNAS; Snoeck 2022 eLife).",
    "Capsicum annuum": "NEGATIVE: no pepper BAK1/SERK3 gene characterised.",
    "Malus domestica":
        "MdBAK1 characterised (Guo et al. 2026 Plant Pathology doi:10.1111/ppa.70132); "
        "apple SERK survey places MdSERK1/8/9 in the AtSERK3-4 clade while MdSERK4 is "
        "the AtSERK1/2 orthologue. No native gene ID published.",
    "Brassica rapa":
        "BrBAK1 characterised (PLoS ONE 2015 doi:10.1371/journal.pone.0142255). "
        "No Bra gene ID published.",
    "Ananas comosus":
        "AcSERK3 named as the best of three pineapple SERK homologues "
        "(Pal & Gangopadhyay 2022 Vegetos 35:571-584). No native gene ID published.",
}


def ensembl_protein(gene_id: str, species: str) -> str:
    url = (f"{ENSEMBL_REST}/sequence/id/{gene_id}?type=protein;species={species};"
           "content-type=application/json;multiple_sequences=1")
    d = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90).read().decode())
    if isinstance(d, dict):
        d = [d]
    return max(d, key=lambda x: len(x["seq"]))["seq"]


def longest_by_gene(fasta_path: str, gene_ids: set[str]) -> dict[str, str]:
    """Pick the longest protein per gene from an Ensembl-style pep FASTA."""
    out: dict[str, str] = {}
    for rec in SeqIO.parse(fasta_path, "fasta"):
        m = re.search(r"gene:(\S+)", rec.description)
        if not m:
            continue
        g = m.group(1)
        key = next((t for t in gene_ids if t.replace(".", "_").upper() == g.upper()), None)
        if key is None:
            continue
        s = str(rec.seq).replace("*", "")
        if len(s) > len(out.get(key, "")):
            out[key] = s
    return out


def download(url: str, dest: str, timeout: int = 600) -> str:
    if os.path.exists(dest):
        log(f"  Already have {os.path.basename(dest)}; skipping download")
        return dest
    log(f"  Downloading {url}")
    t0 = time.time()
    data = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()
    if url.endswith(".gz"):
        data = gzip.decompress(data)
    open(dest, "wb").write(data)
    mb = len(data) / (1024 * 1024)
    log(f"  Saved {os.path.basename(dest)} ({mb:.1f} MB, {time.time() - t0:.0f}s)")
    return dest


def fetch_tier1(workdir: str) -> dict[str, str]:
    """Retrieve each tier-1 protein from its own annotation, not an aggregator."""
    path = os.path.join(workdir, "tier1_seqs.json")
    if os.path.exists(path):
        log(f"Loading cached tier-1 sequences from {path}")
        return json.load(open(path))
    log("Fetching tier-1 literature ground-truth sequences ...")
    seqs: dict[str, str] = {}

    log("  Getting Arabidopsis thaliana BAK1 from Ensembl")
    seqs["AT4G33430"] = ensembl_protein("AT4G33430.1", "arabidopsis_thaliana")
    log("  Getting Oryza sativa SERK genes from Ensembl")
    for gid in ("Os04g0457800", "Os08g0174700"):
        seqs[gid] = ensembl_protein(gid, "oryza_sativa")

    log("  Getting Solanum lycopersicum SERK3 from ITAG4.0")
    itag = download("https://solgenomics.net/ftp/tomato_genome/annotation/"
                    "ITAG4.0_release/ITAG4.0_proteins.fasta",
                    os.path.join(workdir, "itag.faa"))
    tom = {r.id: str(r.seq).replace("*", "") for r in SeqIO.parse(itag, "fasta")}
    for g in ("Solyc10g047140", "Solyc01g104970"):
        hit = sorted(k for k in tom if k.startswith(g))
        seqs[g] = tom[hit[0]]

    log("  Getting Nicotiana benthamiana SERK3 from Niben101")
    nb = download("https://solgenomics.net/ftp/genomes/Nicotiana_benthamiana/"
                  "annotation/Niben101/Niben101_annotation.proteins.fasta.gz",
                  os.path.join(workdir, "niben.faa"))
    nbp = {r.id.split(".")[0]: str(r.seq).replace("*", "") for r in SeqIO.parse(nb, "fasta")}
    for g in ("Niben101Scf11779g01002", "Niben101Scf00279g02022"):
        seqs[g] = nbp[g]

    log("  Getting Glycine max BAK1 from Ensembl Plants")
    gmx = download(ENSEMBL_FTP + "glycine_max/pep/Glycine_max.Glycine_max_v2.1.pep.all.fa.gz",
                   os.path.join(workdir, "gmax.faa"))
    seqs.update(longest_by_gene(gmx, {"Glyma.15G051600", "Glyma.08G180800"}))

    log("  Getting Solanum tuberosum SERK3 from Ensembl Plants")
    pot = download(ENSEMBL_FTP + "solanum_tuberosum/pep/Solanum_tuberosum.SolTub_3.0.pep.all.fa.gz",
                   os.path.join(workdir, "potato.faa"))
    seqs.update(longest_by_gene(pot, {"PGSC0003DMG400017690", "PGSC0003DMG400012594"}))

    json.dump(seqs, open(path, "w"))
    log(f"Saved {len(seqs)} tier-1 sequences to {path}")
    return seqs


# --------------------------------------------------------------------------
# 4. Locus identifier parsing
# --------------------------------------------------------------------------
# Anchored prefix capture. Order matters: accession-style patterns are matched
# before the generic two-letter assembly pattern. Do NOT strip trailing
# suffixes first -- a greedy `_\d+$` rule reduces XP_014499491 to "XP".

NAMESPACE_PATTERNS = [
    ("Araport11",          r"^(AT\d[Gg]\d{5})"),
    ("ITAG tomato",        r"^(Solyc\d+g\d+)"),
    ("SolGenomics Nb",     r"^(Niben\d+Scf\d+g\d+)"),
    ("Phytozome Gmax",     r"^(Glyma\.?\d+G\d+)"),
    ("Phytozome Vunguic",  r"^(Vigun\d+g\d+)"),
    ("Phytozome Vradi",    r"^(Vradi\d+g\d+)"),
    ("Phytozome other",
     r"^(Thhalv\d+m|CsaV3_\d+G\d+|Ccajan_\d+|Distr\d+s\d+|Eusyr\d+s\d+|PHAVU_\d+G?\d*g?"
     r"|Medtr\d+g\d+|Potri\.\d+G\d+|LSAT\d+_V\d+C\d+|ITF\d+g\d+|Lp\d+g\d+)"),
    ("Pepper CM334",       r"^(CA\d\dg\d+)"),
    ("BRAD Brassica",      r"^(Bra(?:ra[A-Z])?\d+)"),
    ("NCBI accession",
     r"^((?:XP|NP|AP|BK|ON|OP|PZ|MW|MN)_?\d{5,}|(?:ABW|ABX|AAO|AAF|KAI|CAA|CAB|BAA|AGL|QQQ)\d+)"),
    ("Ngou2025 custom",    r"^([A-Z]{2}_\d+)"),
]


def parse_locus(raw) -> tuple[str, str]:
    s = str(raw).strip()
    for name, pat in NAMESPACE_PATTERNS:
        m = re.match(pat, s, re.I)
        if m:
            return name, m.group(1)
    return "unmapped", s


SPECIES_ALIAS = {
    "Brasica rapa": "Brassica rapa",
    "Lycopersicum esculentum": "Solanum lycopersicum",
    "Capsicum annum": "Capsicum annuum",
    "Solanum habrochates": "Solanum habrochaites",
    "Solanum microdontum gigantophyllum": "Solanum microdontum",
    "Solanum megistacrolobum subsp. Toralapanum": "Solanum megistacrolobum",
    "Ckanehirae": "Cinnamomum kanehirae",
}
for _g in ("Ajanhuiri Group", "Phureja Group1", "Phureja Group2",
           "Phureja Group3", "Phureja Group4"):
    SPECIES_ALIAS[f"Solanum tuberosum {_g}"] = "Solanum tuberosum"


def normalize_species(label: str) -> str:
    s = str(label).strip()
    return SPECIES_ALIAS.get(s, s)


# --------------------------------------------------------------------------
# 5. Tier 2 - ortholog search in the receptor's own genome
# --------------------------------------------------------------------------
# Candidates are ranked by a CLADE MARGIN, not by raw similarity:
#
#     delta = max identity to tier-1 BAK1-clade queries
#           - max identity to AtSERK1 / AtSERK2
#
# Ranking by raw maximum identity is wrong: it measures how SERK-like a
# protein is, not which clade it belongs to. SERK1/2-clade genes are more
# conserved overall and outrank true BAK1 orthologues -- under that metric ten
# eudicot species received a SERK1/2-clade gene (Populus scored 90.6 vs
# AtSERK1/2 against 79.0 vs the BAK1 clade). Switching to delta changed the
# call in 11 of 32 species.

MIN_LEN, MAX_LEN = 500, 800
MIN_BAK_IDENTITY = 60.0


def ensembl_directories() -> list[str]:
    """NOTE the digits in the character class: assembly-suffixed directories
    such as vigna_unguiculata_gca004118075v2rs are invisible to `[a-z_]+`."""
    idx = urllib.request.urlopen(urllib.request.Request(ENSEMBL_FTP, headers=UA), timeout=180).read().decode()
    return sorted(set(re.findall(r'href="([a-z0-9_]+)/"', idx)))


def resolve_directory(species: str, available: list[str]) -> str | None:
    slug = normalize_species(species).lower().replace(" ", "_").replace(".", "")
    for a in available:
        if a == slug or a.startswith(slug + "_"):
            return a
    return None


def search_proteome(slug: str, queries: list[str], refs: dict[str, str]) -> dict:
    log(f"    Looking up pep file listing for {slug}")
    listing = urllib.request.urlopen(
        urllib.request.Request(ENSEMBL_FTP + slug + "/pep/", headers=UA), timeout=120).read().decode()
    files = re.findall(r'href="([^"]*pep\.all\.fa\.gz)"', listing)
    if not files:
        log(f"    No pep.all.fa.gz file found for {slug}")
        return {"error": "no pep file"}
    log(f"    Downloading {files[0]} (this can take several minutes) ...")
    t0 = time.time()
    raw = urllib.request.urlopen(
        urllib.request.Request(ENSEMBL_FTP + slug + "/pep/" + files[0], headers=UA), timeout=900).read()
    log(f"    Download finished ({len(raw) / (1024 * 1024):.1f} MB in {time.time() - t0:.0f}s). "
        f"Now scanning proteins and aligning ...")
    handle = io.StringIO(gzip.decompress(raw).decode("utf-8", "ignore"))

    s12 = [refs[a] for a in SERK12]
    outg = [refs[a] for a in OUTGROUP]
    hits = []
    n_total = n_len = n_bak = 0
    t_scan = time.time()
    for rec in SeqIO.parse(handle, "fasta"):
        n_total += 1
        if n_total % 5000 == 0:
            log(f"    Still scanning: {n_total} proteins checked "
                f"({n_len} right length, {n_bak} passed BAK1 filter, "
                f"{len(hits)} clade hits so far, {time.time() - t_scan:.0f}s)")
        s = str(rec.seq).replace("*", "")
        if not MIN_LEN <= len(s) <= MAX_LEN:
            continue
        n_len += 1
        bak = max(pid(s, q) for q in queries)
        if bak < MIN_BAK_IDENTITY:
            continue
        n_bak += 1
        b12 = max(pid(s, q) for q in s12)
        bog = max(pid(s, q) for q in outg)
        if bog >= max(bak, b12):          # outgroup: SOBIR1 / NIK1-like
            continue
        g = re.search(r"gene:(\S+)", rec.description)
        hits.append({"gene": g.group(1) if g else rec.id, "protein": rec.id,
                     "bak": round(bak, 1), "s12": round(b12, 1),
                     "delta": round(bak - b12, 1), "seq": s})
    hits.sort(key=lambda h: -h["delta"])
    top = hits[0] if hits else None
    if top:
        log(f"    Finished {slug}: {n_total} proteins scanned. "
            f"Best hit is {top['gene']} (delta={top['delta']}, bak={top['bak']}) "
            f"in {time.time() - t_scan:.0f}s")
    else:
        log(f"    Finished {slug}: {n_total} proteins scanned, no clade hits "
            f"in {time.time() - t_scan:.0f}s")
    return {"assembly": files[0], "hits": hits[:5]}


# --------------------------------------------------------------------------
# 6. Tier 3a - transient expression host
# --------------------------------------------------------------------------
# Ngou et al. 2025 expressed CHIMERAS in N. benthamiana: a subgroup LRR
# ectodomain fused to the BRI1 (Xb) kinase domain, BES1 dephosphorylation
# readout. The co-receptor that engaged is endogenous NbSERK3/BAK1, so this is
# a statement about the assay system, not a substitution.

TRANSIENT_HOSTS = {
    "Ngou_2025": dict(
        gene="Niben101Scf11779g01002",
        alt="Niben101Scf00279g02022 (second Nb homeolog)",
        source="Niben101_annotation.proteins",
        note="receptor assayed as a chimera (LRR-RLK-XII ectodomain fused to the "
             "BRI1/Xb kinase domain) transiently expressed in Nicotiana benthamiana, "
             "BES1 dephosphorylation readout; the operative co-receptor is the "
             "endogenous N. benthamiana SERK3/BAK1, not one from the source species. "
             "Ngou et al. 2025 Science doi:10.1126/science.adx2508"),
}


# --------------------------------------------------------------------------
# 7. Tier 3b - closest ground-truth substitution
# --------------------------------------------------------------------------
# Donor pool is restricted to species with a literature-validated co-receptor,
# so every substituted sequence traces to a characterised gene. Distance is the
# number of leading NCBI Taxonomy lineage ranks shared from the root.
#
# MIN_SHARED_RANKS keeps out meaningless matches: without it, Ginkgo and Pinus
# pair with a legume at 9 shared ranks (Spermatophyta = "all seed plants") and
# Nelumbo at 11 (Mesangiospermae = "all flowering plants").

MIN_SHARED_RANKS = 13


def ncbi_lineage(species: str, cache: dict, retries: int = 6) -> list[str] | None:
    """NCBI caps anonymous use at ~3 requests/sec and returns errors when
    exceeded. Transient failures must be retried and must NOT be cached, or a
    rate-limited burst permanently poisons the cache -- in one run this
    silently dropped 40 of 70 species, including donor species, and sent 313
    rows to NA. Set NCBI_API_KEY to raise the limit to 10 req/sec."""
    if species in cache and cache[species]:
        return cache[species]
    key = os.environ.get("NCBI_API_KEY")
    delay = 0.4 if key else 1.1
    log(f"Looking up NCBI taxonomy for {species}")
    for attempt in range(retries):
        try:
            p = {"db": "taxonomy", "term": species, "retmode": "json"}
            if key:
                p["api_key"] = key
            ids = json.loads(urllib.request.urlopen(
                urllib.request.Request(EUTILS + "esearch.fcgi?" + urllib.parse.urlencode(p),
                                       headers=UA), timeout=60
            ).read().decode())["esearchresult"]["idlist"]
            if not ids:
                cache[species] = None          # genuine absence: cache it
                log(f"  No NCBI taxonomy hit for {species}")
                return None
            time.sleep(delay)
            p = {"db": "taxonomy", "id": ids[0], "retmode": "xml"}
            if key:
                p["api_key"] = key
            xml = urllib.request.urlopen(urllib.request.Request(
                EUTILS + "efetch.fcgi?" + urllib.parse.urlencode(p),
                headers=UA), timeout=60).read().decode()
            lin = re.search(r"<Lineage>(.*?)</Lineage>", xml, re.S)
            sci = re.search(r"<ScientificName>(.*?)</ScientificName>", xml, re.S)
            if not lin:
                raise ValueError("no lineage in response")
            cache[species] = lin.group(1).split("; ") + ([sci.group(1)] if sci else [])
            time.sleep(delay)
            return cache[species]
        except Exception as e:
            log(f"  Taxonomy lookup attempt {attempt + 1}/{retries} failed: {e}")
            time.sleep(delay * (2 ** attempt))  # backoff; do NOT cache
    raise RuntimeError(f"NCBI taxonomy lookup failed after {retries} attempts: {species}")


def shared_ranks(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def choose_donor(species, lineage_cache, donors):
    """Closest ground-truth donor.

    Primary key is shared lineage depth. Ties are broken by *residual donor
    depth* -- how many ranks the donor itself sits below the shared node --
    preferring the donor nearer that node. This matters for wild potato
    relatives: S. tuberosum and S. lycopersicum both share the lineage only as
    far as `Solanum`, but S. lycopersicum carries an extra rank
    (`Solanum subgen. Lycopersicon`) beneath it, so S. tuberosum is the closer
    donor. Tomato-clade recipients are unaffected because they share that
    subgenus rank and so win on the primary key.
    """
    lin = ncbi_lineage(normalize_species(species), lineage_cache) or []
    if not lin:
        return None

    def key(d):
        dl = ncbi_lineage(d, lineage_cache) or []
        n = shared_ranks(lin, dl)
        return (-n, len(dl) - n, d)

    best = sorted(donors, key=key)[0]
    n = shared_ranks(lin, ncbi_lineage(best, lineage_cache) or [])
    if n == 0:
        return None
    # Below the floor the substitution carries little orthology information
    # (9 ranks = Spermatophyta, 11 = Mesangiospermae). It is still returned so
    # no row is left empty, but flagged so it can be filtered out.
    return {"donor": best, "shared": n, "rank": lin[n - 1],
            "below_floor": n < MIN_SHARED_RANKS}


def validate_donor_clade(seq: str, refs: dict[str, str], species: str) -> bool:
    """Reject a donor whose own gene is SERK1/2-clade. All three Citrus
    clementina candidates failed this, so Rutaceae recipients fall through."""
    basal = {"Oryza sativa", "Brachypodium distachyon", "Ananas comosus", "Papaver somniferum"}
    bak = max(pid(seq, refs[a]) for a in BAK1_CLADE)
    s12 = max(pid(seq, refs[a]) for a in SERK12)
    return not (s12 > bak and species not in basal)


# --------------------------------------------------------------------------
# 8. Orchestration
# --------------------------------------------------------------------------

def resolve_path(path: str | Path) -> Path:
    """Absolute paths stay as-is; relative paths are resolved from the repo root."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Assign BAK1/SERK3-clade co-receptors to LRR-PRR / ligand records.")
    ap.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help=f"Excel workbook of receptor/ligand records "
             f"(default: {DEFAULT_INPUT.relative_to(REPO_ROOT)})",
    )
    ap.add_argument(
        "--workdir",
        default=str(DEFAULT_WORKDIR),
        help=f"Cache directory for downloads and intermediate JSON "
             f"(default: {DEFAULT_WORKDIR.relative_to(REPO_ROOT)})",
    )
    ap.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"Output TSV path "
             f"(default: {DEFAULT_OUT.relative_to(REPO_ROOT)})",
    )
    args = ap.parse_args()

    input_path = resolve_path(args.input)
    workdir = resolve_path(args.workdir)
    out_path = resolve_path(args.out)

    if not input_path.is_file():
        raise SystemExit(f"input not found: {input_path}")

    workdir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    workdir_s = str(workdir)

    t_run = time.time()
    log("=" * 60)
    log("SERK co-receptor pipeline starting")
    log(f"Input:   {input_path}")
    log(f"Workdir: {workdir}")
    log(f"Output:  {out_path}")
    log("=" * 60)

    log("")
    log("Step 1/4: Loading reference and tier-1 sequences")
    refs = fetch_refs(workdir_s)
    tier1_seqs = fetch_tier1(workdir_s)
    queries = list(tier1_seqs.values())
    log("Step 1/4 complete")

    log("")
    log("Step 2/4: Reading input Excel workbook")
    xl = pd.ExcelFile(input_path)
    df = pd.concat([xl.parse(s).dropna(how="all").assign(Source_sheet=s)
                    for s in xl.sheet_names], ignore_index=True)
    df[["ns", "base"]] = df["Locus ID/Genbank"].apply(lambda x: pd.Series(parse_locus(x)))
    df["species_norm"] = df["Plant species"].apply(normalize_species)
    n_sp = df.species_norm.nunique()
    log(f"Loaded {len(df)} rows from {len(xl.sheet_names)} sheets "
        f"({n_sp} unique plant species)")
    log("Step 2/4 complete")

    # -- tier 2: species needing their own genome searched
    log("")
    log("Step 3/4: Searching proteomes for SERK orthologs "
        "(this is the slow step — downloads + alignments)")
    ortho_path = workdir / "ortho.json"
    ortho = json.load(open(ortho_path)) if ortho_path.exists() else {}
    if ortho:
        log(f"Found existing ortho cache with {len(ortho)} species at {ortho_path}")
    log("Listing available Ensembl Plants directories ...")
    available = ensembl_directories()
    log(f"Found {len(available)} Ensembl plant directories")
    need = sorted({s for s in df.species_norm.unique() if s not in TIER1})
    todo = [s for s in need if s not in ortho]
    log(f"{len(need)} species need an ortholog check; "
        f"{len(todo)} still to search ({len(need) - len(todo)} already cached)")
    for i, sp in enumerate(need, 1):
        if sp in ortho:
            err = ortho[sp].get("error")
            hit = (ortho[sp].get("hits") or [None])[0]
            if err:
                status = f"cached error ({err})"
            elif hit:
                status = f"cached hit {hit['gene']} (delta={hit['delta']})"
            else:
                status = "cached, no hits"
            log(f"  [{i}/{len(need)}] {sp}: {status}")
            continue
        slug = resolve_directory(sp, available)
        if not slug:
            log(f"  [{i}/{len(need)}] {sp}: no Ensembl proteome available")
            ortho[sp] = {"error": "no proteome"}
        else:
            log(f"  [{i}/{len(need)}] Searching {sp} (Ensembl slug: {slug})")
            try:
                ortho[sp] = search_proteome(slug, queries, refs)
            except Exception as e:
                log(f"    ERROR while searching {sp}: {e}")
                ortho[sp] = {"error": str(e)[:80]}
        json.dump(ortho, open(ortho_path, "w"))
        log(f"    Saved progress to ortho.json ({i}/{len(need)} species done)")
    log("Step 3/4 complete")

    # -- tier 3b: donor pool = tier-1 species only
    log("")
    log("Step 4/4: Assigning a co-receptor to every input row")
    lineage_cache_path = workdir / "tax.json"
    lineage_cache = json.load(open(lineage_cache_path)) if lineage_cache_path.exists() else {}
    if lineage_cache:
        log(f"Loaded taxonomy cache with {len(lineage_cache)} species")
    donors = [s for s in TIER1 if validate_donor_clade(tier1_seqs[TIER1[s]["native"]], refs, s)]
    log(f"Using {len(donors)} literature-validated donor species: {', '.join(donors)}")

    rows = []
    n_rows = len(df)
    progress_every = max(1, n_rows // 20)  # ~5% ticks
    for i, r in df.iterrows():
        n_done = len(rows) + 1
        sp, ns = r.species_norm, r.ns
        study = str(r.get("Literature_Data", ""))
        rec = dict(
            Source_sheet=r["Source_sheet"],
            row_id=i + 1, Plant_species=r["Plant species"], Species_normalized=sp,
            Receptor=r["Receptor"], Receptor_type=r["Receptor Type"],
            Receptor_locus_ID=r["Locus ID/Genbank"], Receptor_annotation=ns,
            Receptor_Sequence=r["Receptor Sequence"], Ligand=r.get("Ligand"),
            Ligand_Sequence=r.get("Ligand Sequence"), Immunogenicity=r.get("Immunogenicity"),
            Coreceptor="BAK1/SERK3 clade",
            Requires_SOBIR1="yes" if str(r["Receptor Type"]).strip() == "LRR-RLP" else "no",
            Literature_note=LIT_NOTES.get(sp, ""),
        )
        host = next((h for k, h in TRANSIENT_HOSTS.items() if k in study), None)

        if sp in TIER1:                                       # tier 1
            t = TIER1[sp]
            rec.update(Tier="1 - literature ground truth", Published_SERK=t["gene"],
                       Published_SERK_reference=t["ref"], SERK_ID=t["native"],
                       SERK_Sequence=tier1_seqs[t["native"]], Alternate_SERK_ID=t["alt"],
                       SERK_source_genome=t["source"], Clade_margin_vs_AtSERK1_2="",
                       Bridge_method=t["bridge"], Substitution_basis="none - own species")
        elif host:                                            # tier 3a
            rec.update(Tier="3a - transient expression host", Published_SERK="NbSerk3A/NbSerk3B",
                       Published_SERK_reference=host["note"], SERK_ID=host["gene"],
                       SERK_Sequence=tier1_seqs[host["gene"]], Alternate_SERK_ID=host["alt"],
                       SERK_source_genome=host["source"], Clade_margin_vs_AtSERK1_2="",
                       Bridge_method=host["note"],
                       Substitution_basis="none - host-species co-receptor, not a substitution")
        elif ortho.get(sp, {}).get("hits") and ortho[sp]["hits"][0]["delta"] > 0:   # tier 2
            h = ortho[sp]["hits"][0]
            rec.update(Tier="2 - ortholog from genome, anchored on tier-1 literature sequences",
                       Published_SERK="no SERK characterised in this species",
                       Published_SERK_reference="NA", SERK_ID=h["gene"], SERK_Sequence=h["seq"],
                       Alternate_SERK_ID=";".join(x["gene"] for x in ortho[sp]["hits"][1:]
                                                  if x["delta"] > 0) or "none",
                       SERK_source_genome=ortho[sp]["assembly"],
                       Clade_margin_vs_AtSERK1_2=h["delta"],
                       Bridge_method="ortholog search: tier-1 SERK proteins as queries vs this "
                                     "genome proteome; SOBIR1/NIK1 negative controls; ranked by "
                                     "identity to BAK1 clade MINUS identity to AtSERK1/2",
                       Substitution_basis="none - own genome")
        else:                                                 # tier 3b or NA
            d = choose_donor(sp, lineage_cache, donors)
            if d:
                t = TIER1[d["donor"]]
                far = d.get("below_floor")
                rec.update(Tier=("3c - distant substitution, below confidence floor"
                                 if far else "3b - closest ground-truth substitution"),
                           Published_SERK="no SERK characterised in this species",
                           Published_SERK_reference=t["ref"], SERK_ID=t["native"],
                           SERK_Sequence=tier1_seqs[t["native"]], Alternate_SERK_ID="none",
                           SERK_source_genome=t["source"], Clade_margin_vs_AtSERK1_2="",
                           Substitution_basis=f"{d['donor']} | shared to {d['rank']} "
                                              f"({d['shared']} shared ranks) | literature-validated donor",
                           Bridge_method=(
                               f"closest ground-truth substitution: no genome for {sp}; SERK taken "
                               f"from {d['donor']}, the taxonomically closest species with a "
                               f"literature-validated co-receptor (shared to {d['rank']}, "
                               f"{d['shared']} ranks)"
                               + (f". BELOW CONFIDENCE FLOOR ({MIN_SHARED_RANKS} ranks): "
                                  f"'{d['rank']}' is too broad to carry orthology information, so "
                                  f"this is a placeholder rather than a supported assignment."
                                  if far else "")))
            else:
                rec.update(Tier="NA", Published_SERK="NA", Published_SERK_reference="NA",
                           SERK_ID="NA", SERK_Sequence="NA", Alternate_SERK_ID="NA",
                           SERK_source_genome="NA", Clade_margin_vs_AtSERK1_2="",
                           Substitution_basis="none",
                           Bridge_method="no suitable donor: nearest available genome is more "
                                         f"distant than the {MIN_SHARED_RANKS}-rank floor")
        rows.append(rec)
        if n_done == 1 or n_done % progress_every == 0 or n_done == n_rows:
            log(f"  Assigned {n_done}/{n_rows} rows "
                f"(latest: {sp} -> {rec['SERK_ID']} [{rec['Tier'].split(' - ')[0]}])")

    json.dump(lineage_cache, open(lineage_cache_path, "w"))
    out = pd.DataFrame(rows)
    out.to_csv(out_path, sep="\t", index=False)
    log("")
    log("=" * 60)
    log(f"Done. Wrote results to {out_path}")
    log(f"{len(out)} rows total; {(out.SERK_ID != 'NA').sum()} assigned; "
        f"{out[out.SERK_ID != 'NA'].SERK_ID.nunique()} distinct co-receptor genes")
    log(f"Total runtime: {time.time() - t_run:.0f}s")
    log("Tier counts:")
    print(out.Tier.value_counts().to_string(), flush=True)
    log("=" * 60)


if __name__ == "__main__":
    main()

from pathlib import Path
import re
import shutil
import sys
import os

# Add both the project root and LRR_Annotation directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "LRR_Annotation"))

from LRR_Annotation.geom_lrr import Loader, Analyzer, Plotter
from LRR_Annotation.extract_lrr_sequences import LRRSequenceExtractor


def parse_alphafold_log(log_file):
    """
    Parse AlphaFold log file to extract final pLDDT and pTM scores for each model.
    
    Args:
        log_file (Path): Path to the log.txt file
        
    Returns:
        dict: Dictionary with results organized by receptor and model
    """
    # ... existing code ...
    results = {}
    current_receptor = None
    
    with open(log_file, 'r') as f:
        for line in f:
            if "Query" in line and "length" in line:
                current_receptor = line.split("Query")[1].split(":")[1].split("(")[0].strip()
                results[current_receptor] = {}
                continue
                
            if "alphafold2_ptm_model_" in line and "took" not in line:
                model_match = re.search(r'model_(\d+)', line)
                if not model_match:
                    continue
                    
                model_num = model_match.group(1)
                
                plddt_match = re.search(r'pLDDT=(\d+\.?\d*)', line)
                ptm_match = re.search(r'pTM=(\d+\.?\d*)', line)
                
                if plddt_match and ptm_match:
                    plddt = float(plddt_match.group(1))
                    ptm = float(ptm_match.group(1))
                    
                    if model_num not in results[current_receptor] or \
                       plddt > results[current_receptor][model_num]['plddt']:
                        results[current_receptor][model_num] = {
                            'plddt': plddt,
                            'ptm': ptm
                        }
    
    return results

def write_results(results, output_file):
    """Write results to a clear text file showing only the best model."""
    receptor_width = 100
    
    with open(output_file, 'w') as f:
        f.write("Receptor".ljust(receptor_width))
        f.write("Best Model".ljust(15))
        f.write("pLDDT".ljust(10))
        f.write("pTM\n")
        f.write("-" * (receptor_width + 25) + "\n")
        
        for receptor in sorted(results.keys()):
            models = results[receptor]
            best_model = max(models.items(), key=lambda x: x[1]['plddt'])
            model_num = best_model[0]
            scores = best_model[1]
            
            f.write(f"{receptor.ljust(receptor_width)}")
            f.write(f"model_{model_num}".ljust(15))
            f.write(f"{scores['plddt']:.1f}".ljust(10))
            f.write(f"{scores['ptm']:.3f}\n")
    
    return {receptor: max(models.items(), key=lambda x: x[1]['plddt'])[0] 
            for receptor, models in results.items()}

def copy_best_models(best_models, source_dir, target_dir):
    """
    Copy the best model PDB files to the target directory.
    
    Args:
        best_models (dict): Dictionary mapping receptor names to their best model numbers
        source_dir (Path): Directory containing AlphaFold output
        target_dir (Path): Directory to copy best models to
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    
    for receptor, model_num in best_models.items():
        # Construct the file pattern for glob
        file_pattern = f"{receptor}_unrelaxed_rank_*_alphafold2_ptm_model_{model_num}_seed_*.pdb"
        matching_files = list(source_dir.glob(file_pattern))
        
        if not matching_files:
            print(f"Warning: Could not find model file for {receptor} matching pattern {file_pattern}")
            continue
            
        if len(matching_files) > 1:
            print(f"Warning: Multiple matching files found for {receptor}, using first match")
            
        source_path = matching_files[0]
        target_path = target_dir / f"{receptor}.pdb"
        shutil.copy2(source_path, target_path)
        print(f"Copied {source_path} to {target_path}")

def run_lrr_annotation(pdb_directory):
    """Run LRR annotation on the PDB files."""
    # Initialize objects
    L = Loader()
    A = Analyzer()
    P = Plotter()
    sequence_extractor = LRRSequenceExtractor()

    # Load structures
    L.load_batch(str(pdb_directory))
    
    # Analyze geometry
    A.load_structures(L.structures)
    A.compute_windings()
    A.compute_regressions()

    # Extract LRR sequences
    output_file = Path('./out_data/lrr_annotation_results.txt')
    
    with open(output_file, 'w') as f:
        f.write("PDB_Filename\tRegion_Number\tStart_Position\tEnd_Position\tSequence_Length\tFull_Sequence_Length\tTotal_LRR_Regions\tSequence\n")
    
    for pdb_id, breakpoints in A.breakpoints.items():
        pdb_file = pdb_directory / f"{pdb_id}.pdb"
        pdb_filename = pdb_file.name
        
        if not pdb_file.exists():
            print(f"Warning: PDB file not found for {pdb_id}")
            continue
            
        results = sequence_extractor.analyze_lrr_regions(str(pdb_file), breakpoints)
        
        with open(output_file, 'a') as f:
            for i, (seq, (start, end)) in enumerate(zip(results['lrr_sequences'], 
                                                       results['lrr_positions'])):
                f.write(f"{pdb_filename}\t{i+1}\t{start}\t{end}\t{len(seq)}\t{results['sequence_length']}\t{results['num_lrr_regions']}\t{seq}\n")

    # Cache data
    cache_dir = Path('./LRR_Annotation/cache')
    cache_dir.mkdir(parents=True, exist_ok=True)
    L.cache(str(cache_dir))
    A.cache_geometry(str(cache_dir))
    A.cache_regressions(str(cache_dir))

    # Generate plots
    P.load(A.windings, A.breakpoints, A.slopes)
    plot_dir = Path('./out_data/lrr_annotation_plots')
    plot_dir.mkdir(parents=True, exist_ok=True)
    P.plot_regressions(save=True, directory=str(plot_dir))

def main():
    # Set up paths
    project_root = Path(__file__).parent.parent
    log_file = project_root / "structural_models" / "receptor_only" / "log.txt"
    scores_file = project_root / "results" / "alphafold_scores.txt"
    
    # Source and target directories for PDB files
    source_dir = project_root / "structural_models" / "receptor_only"
    target_dir = project_root / "out_data" / "modeled_structures" / "pdb_for_lrr_annotator"
    
    if not log_file.exists():
        print(f"Error: Log file not found at {log_file}")
        return
        
    # Step 1: Parse AlphaFold results and get best models
    results = parse_alphafold_log(log_file)
    best_models = write_results(results, scores_file)
    print(f"AlphaFold scores written to {scores_file}")
    
    # Step 2: Copy best models to target directory
    copy_best_models(best_models, source_dir, target_dir)
    
    # Step 3: Run LRR annotation
    run_lrr_annotation(target_dir)
    print("LRR annotation completed")

if __name__ == '__main__':
    main() 
from pathlib import Path
import re

def parse_alphafold_log(log_file):
    """
    Parse AlphaFold log file to extract final pLDDT and pTM scores for each model.
    
    Args:
        log_file (Path): Path to the log.txt file
        
    Returns:
        dict: Dictionary with results organized by receptor and model
    """
    results = {}
    current_receptor = None
    
    with open(log_file, 'r') as f:
        for line in f:
            # Find receptor name - get the full name without splitting
            if "Query" in line and "length" in line:
                # Extract everything between "Query X/Y:" and "(length Z)"
                current_receptor = line.split("Query")[1].split(":")[1].split("(")[0].strip()
                results[current_receptor] = {}
                continue
                
            # Find model results - only get the final recycle for each model
            if "alphafold2_ptm_model_" in line and "took" not in line:
                # Extract model number using regex
                model_match = re.search(r'model_(\d+)', line)
                if not model_match:
                    continue
                    
                model_num = model_match.group(1)
                
                # Extract pLDDT and pTM scores using regex
                plddt_match = re.search(r'pLDDT=(\d+\.?\d*)', line)
                ptm_match = re.search(r'pTM=(\d+\.?\d*)', line)
                
                if plddt_match and ptm_match:
                    plddt = float(plddt_match.group(1))
                    ptm = float(ptm_match.group(1))
                    
                    # Only update if it's a new model or has higher scores
                    if model_num not in results[current_receptor] or \
                       plddt > results[current_receptor][model_num]['plddt']:
                        results[current_receptor][model_num] = {
                            'plddt': plddt,
                            'ptm': ptm
                        }
    
    return results

def write_results(results, output_file):
    """
    Write results to a clear text file in a three-column format showing only the best model.
    
    Args:
        results (dict): Dictionary containing the results
        output_file (Path): Path to output file
    """
    # Increase column width for receptor names
    receptor_width = 100  # Increased from 60 to 100
    
    with open(output_file, 'w') as f:
        # Write header
        f.write("Receptor".ljust(receptor_width))
        f.write("Best Model".ljust(15))
        f.write("pLDDT".ljust(10))
        f.write("pTM\n")
        f.write("-" * (receptor_width + 25) + "\n")  # Adjusted line length
        
        # Sort receptors alphabetically and write their best models
        for receptor in sorted(results.keys()):
            models = results[receptor]
            best_model = max(models.items(), key=lambda x: x[1]['plddt'])
            model_num = best_model[0]
            scores = best_model[1]
            
            f.write(f"{receptor.ljust(receptor_width)}")  # Removed truncation
            f.write(f"model_{model_num}".ljust(15))
            f.write(f"{scores['plddt']:.1f}".ljust(10))
            f.write(f"{scores['ptm']:.3f}\n")

def main():
    project_root = Path(__file__).parent.parent
    log_file = project_root / "structural_models" / "receptor_only" / "log.txt"
    output_file = project_root / "results" / "alphafold_scores.txt"
    
    if not log_file.exists():
        print(f"Error: Log file not found at {log_file}")
        return
        
    results = parse_alphafold_log(log_file)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_results(results, output_file)
    print(f"Results written to {output_file}")

if __name__ == '__main__':
    main()
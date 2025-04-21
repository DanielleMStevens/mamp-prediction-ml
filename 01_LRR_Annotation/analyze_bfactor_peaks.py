import os
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm
import pickle # Added for potential loading errors

# Assuming geom_lrr is in the same parent directory or Python path
try:
    from geom_lrr.loader import Loader
    # Import Analyzer and compute_winding function
    from geom_lrr.analyzer import Analyzer, compute_winding
except ImportError:
    print("Error: Could not import Loader, Analyzer, or compute_winding from geom_lrr.")
    print("Make sure 'geom_lrr' directory is accessible and contains analyzer.py with compute_winding.")
    exit() # Or handle the error appropriately

def analyze_lrr_bfactor_peaks(pdb_dir, cache_dir, period=25, filter_order=10):
    """
    Analyzes B-factor peaks within LRR regions defined by cached breakpoints.
    Computes winding numbers on the fly from loaded structures.
    Loads structures via load_batch and B-factors via load_single (inefficient).

    Args:
        pdb_dir (str): Path to the directory containing PDB files.
        cache_dir (str): Path to the directory containing cached regression results
                         (breakpoints.pickle).
        period (int, optional): Approximate period for Butterworth filter. Defaults to 25.
        filter_order (int, optional): Order for Butterworth filter. Defaults to 10.

    Returns:
        pandas.DataFrame: A DataFrame containing peak information with columns:
                          'Protein Key', 'Residue Index', 'Filtered B-Factor', 'Winding Number'.
                          Returns an empty DataFrame if essential data is missing.
    """
    print(f"Loading Structures via load_batch from PDBs in: {pdb_dir}")
    loader = Loader()
    keys_loaded = []
    try:
        # Use load_batch to get structures and the list of keys
        # Note: Original load_batch only populates self.structures
        loader.load_batch(pdb_dir, progress=True)
        structures = loader.structures
        if not structures:
            print("Warning: No structures loaded via load_batch. Check PDB directory and files.")
            return pd.DataFrame()
        keys_loaded = list(structures.keys()) # Get keys from successfully loaded structures
        print(f"Found {len(keys_loaded)} structures via load_batch.")

    except FileNotFoundError:
        print(f"Error: PDB directory not found: {pdb_dir}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error during load_batch for structures: {e}")
        return pd.DataFrame()

    # Now, load B-factors individually using load_single for the keys found
    print(f"Loading B-factors via load_single for {len(keys_loaded)} keys...")
    bfactors = {} # Initialize bfactors dictionary
    for key in tqdm(keys_loaded, desc="Loading B-factors"):
        try:
            # Temporarily assign to loader's dict, then copy, to use existing method
            loader.bfactors = {} # Clear before each call if load_single appends/updates
            loader.load_single(pdb_dir, f"{key}.pdb") # Assumes key is filename without .pdb
            if key in loader.bfactors:
                 bfactors[key] = loader.bfactors[key] # Copy loaded bfactor
            # load_single prints its own warnings on failure based on original code
        except Exception as e:
            # Catch any unexpected error during load_single
            print(f"\nError calling load_single for {key}.pdb: {e}")
            continue # Skip to next key

    # Clear loader's bfactor dict as we have our own copy now
    loader.bfactors = {}
    print(f"Finished loading B-factors. Found B-factors for {len(bfactors)} keys.")

    if not bfactors:
         print("Error: Failed to load any B-factors using load_single.")
         return pd.DataFrame()

    # --- Load Cached Breakpoints ---
    print(f"Loading cached regression data (breakpoints) from: {cache_dir}")
    analyzer = Analyzer()
    try:
        analyzer.retrieve_regressions(cache_dir) # Loads breakpoints, slopes
    except (FileNotFoundError, pickle.UnpicklingError) as e:
        print(f"Error: Cache directory or essential regression files not found/corrupt in {cache_dir}: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading cached regression data: {e}")
        return pd.DataFrame()

    # Load the structures and B-factors obtained above into the analyzer
    analyzer.load_structures(structures)
    analyzer.load_bfactors(bfactors) # Load the bfactors collected via load_single

    results = []
    required_data = {'breakpoints', 'structures', 'bfactors'}
    if not required_data.issubset(analyzer.__dict__.keys()) or \
       not analyzer.breakpoints or not analyzer.structures or not analyzer.bfactors:
        # Check counts as well
        print(f"Error: Analyzer is missing essential data.")
        print(f"Breakpoints: {len(analyzer.breakpoints)}, Structures: {len(analyzer.structures)}, B-factors: {len(analyzer.bfactors)}")
        return pd.DataFrame()

    # --- Filter Setup ---
    nyquist = 0.5
    low_cutoff = 0.5 / period
    high_cutoff = 2.0 / period
    if high_cutoff >= nyquist:
        print(f"Warning: High cutoff frequency ({high_cutoff}) >= Nyquist ({nyquist}). Adjusting.")
        high_cutoff = nyquist * 0.99
    if low_cutoff >= high_cutoff:
        print(f"Error: Low cutoff ({low_cutoff}) >= high cutoff ({high_cutoff}). Cannot create filter.")
        return pd.DataFrame()
    try:
        sos = signal.butter(filter_order, [low_cutoff, high_cutoff], 'bandpass', output='sos', fs=1.0)
    except ValueError as e:
        print(f"Error creating Butterworth filter: {e}")
        return pd.DataFrame()
    # --- End Filter Setup ---

    print("Analyzing B-factor peaks in LRR regions (computing windings)...")
    # Iterate through proteins that have BOTH structure and bfactor data loaded AND have breakpoint data
    # Intersect keys from all three sources
    valid_keys = set(analyzer.structures.keys()) & set(analyzer.bfactors.keys()) & set(analyzer.breakpoints.keys())
    print(f"Processing {len(valid_keys)} proteins with structures, b-factors, and breakpoints.")

    if not valid_keys:
         print("No proteins found with all required data (structure, b-factor, breakpoints).")
         return pd.DataFrame()

    for key in tqdm(list(valid_keys), desc="Processing Proteins"):
        structure = analyzer.structures[key]
        bfactor = analyzer.bfactors[key]
        breakpoints = analyzer.breakpoints[key]

        # Compute winding number
        try:
            winding_res = compute_winding(structure)
            winding = winding_res["winding"]
        except Exception as e:
            print(f"Warning: Skipping {key}, failed to compute winding number: {e}")
            continue

        # Basic check: winding array is typically N-1, bfactor N.
        if len(bfactor) != len(winding) + 1:
             print(f"Warning: Skipping {key}, length mismatch between B-factor ({len(bfactor)}) and computed winding ({len(winding)}).")
             continue

        if len(breakpoints) < 2:
            continue # Need at least one segment

        # Process each LRR segment
        for j in range(0, len(breakpoints) - 1, 2):
            a = breakpoints[j]
            b = breakpoints[j+1]

            if a < 0 or b > len(bfactor) or a >= b:
                 print(f"Warning: Skipping segment [{a}, {b}) for {key} due to invalid indices (B-factor length: {len(bfactor)}).")
                 continue

            bfactor_segment = bfactor[a:b]

            min_len_for_filter = 3
            if len(bfactor_segment) < min_len_for_filter:
                continue

            try:
                bff = signal.sosfiltfilt(sos, bfactor_segment)
            except ValueError as e:
                 print(f"Warning: Could not filter segment [{a}, {b}) for {key}: {e}")
                 continue

            # Normalize
            bff_max = np.max(np.abs(bff))
            if bff_max > 1e-9:
                bff /= bff_max
            else:
                bff[:] = 0

            # Find peaks
            idx = np.arange(1, bff.size - 1)
            peaks = idx[(bff[idx] > bff[idx - 1]) & (bff[idx] > bff[idx + 1])]

            # Record peak information
            for p in peaks:
                residue_idx = a + p
                filtered_b_val = bff[p]
                winding_idx = residue_idx - 1
                if 0 <= winding_idx < len(winding):
                    winding_val = winding[winding_idx]
                else:
                    winding_val = np.nan

                results.append({
                    'Protein Key': key,
                    'Residue Index': residue_idx,
                    'Filtered B-Factor': filtered_b_val,
                    'Winding Number': winding_val
                })

    if not results:
        print("No peaks found or processed.")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    return df

if __name__ == "__main__":
    # --- Configuration ---
    PDB_DIRECTORY = "./03_out_data/modeled_structures/pdb_for_lrr_annotator/"
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    CACHE_DIRECTORY = os.path.join(SCRIPT_DIR, "cache")
    # --- End Configuration ---

    if not os.path.isdir(PDB_DIRECTORY):
        print(f"Error: PDB Directory not found: {PDB_DIRECTORY}")
        exit()
    if not os.path.isdir(CACHE_DIRECTORY):
         print(f"Error: Cache Directory not found: {CACHE_DIRECTORY}")
         exit()
    else:
        peak_data = analyze_lrr_bfactor_peaks(PDB_DIRECTORY, CACHE_DIRECTORY)

        if not peak_data.empty:
            print("\n--- B-Factor Peak Analysis Results ---")
            #pd.set_option('display.max_rows', 100)
            #pd.set_option('display.max_columns', 10)
            #pd.set_option('display.width', 120)
            #print(peak_data)
            # Optional: Save results
            output_csv = "bfactor_peak_analysis_computed_winding_workaround.csv"
            peak_data.to_csv(output_csv, index=False)
            print(f"\nResults saved to {output_csv}")
        else:
            print("\nAnalysis finished, but no peak data was generated.")

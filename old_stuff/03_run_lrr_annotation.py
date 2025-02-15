import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geom_lrr import Loader, Analyzer, Plotter
from LRR_Annotation.extract_lrr_sequences import LRRSequenceExtractor

def main():
    # Initialize objects
    L = Loader()
    A = Analyzer()
    P = Plotter()
    sequence_extractor = LRRSequenceExtractor()

    # Cache settings
    pdb_cached = False
    geometry_cached = False
    regressions_cached = False
    make_cache = True

    # Store the input directory for later use
    pdb_directory = './out_data/modeled_structures/pdb_for_lrr_annotator/'

    # Load or retrieve data
    if pdb_cached:
        L.retrieve('./LRR_Annotation/cache')
    else:
        # Changed input directory to use modeled structures
        L.load_batch(pdb_directory)
        
    if geometry_cached:
        A.retrieve_geometry('./LRR_Annotation/cache')
    else:
        A.load_structures(L.structures)
        A.compute_windings()
        
    if regressions_cached:
        A.retrieve_regressions('./LRR_Annotation/cache')
    else:
        A.compute_regressions()

    # Extract LRR sequences using breakpoints from regression analysis
    # Create output file
    output_file = './out_data/lrr_annotation_results.txt'
    
    # Write header to the output file
    with open(output_file, 'w') as f:
        f.write("PDB_Filename\tRegion_Number\tStart_Position\tEnd_Position\tSequence_Length\tFull_Sequence_Length\tTotal_LRR_Regions\tSequence\n")
    
    for pdb_id, breakpoints in A.breakpoints.items():
        # Reconstruct the PDB file path and get filename
        pdb_file = os.path.join(pdb_directory, f"{pdb_id}.pdb")
        pdb_filename = os.path.basename(pdb_file)  # This will get just the filename part
        
        # Check if file exists
        if not os.path.exists(pdb_file):
            print(f"Warning: PDB file not found for {pdb_id}")
            continue
            
        results = sequence_extractor.analyze_lrr_regions(pdb_file, breakpoints)
        
        # Append results to the output file
        with open(output_file, 'a') as f:
            for i, (seq, (start, end)) in enumerate(zip(results['lrr_sequences'], 
                                                       results['lrr_positions'])):
                f.write(f"{pdb_filename}\t{i+1}\t{start}\t{end}\t{len(seq)}\t{results['sequence_length']}\t{results['num_lrr_regions']}\t{seq}\n")

    # Cache data if requested
    if make_cache:
        L.cache('./LRR_Annotation/cache')
        A.cache_geometry('./LRR_Annotation/cache')
        A.cache_regressions('./LRR_Annotation/cache')

    # Load data into plotter
    P.load(A.windings, A.breakpoints, A.slopes)

    # Generate and save plots
    P.plot_regressions(save=True, directory='./out_data/lrr_annotation_plots')

if __name__ == "__main__":
    main() 
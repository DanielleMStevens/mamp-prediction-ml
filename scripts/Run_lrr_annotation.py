from geom_lrr import Loader, Analyzer, Plotter
from extract_lrr_sequences import LRRSequenceExtractor
import os

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
    for pdb_id, breakpoints in A.breakpoints.items():
        # Reconstruct the PDB file path
        pdb_file = os.path.join(pdb_directory, f"{pdb_id}.pdb")
        
        # Check if file exists
        if not os.path.exists(pdb_file):
            print(f"Warning: PDB file not found for {pdb_id}")
            continue
            
        results = sequence_extractor.analyze_lrr_regions(pdb_file, breakpoints)
        
        # Save or print the results
        print(f"\nResults for {pdb_id}:")
        print(f"Full sequence length: {results['sequence_length']}")
        print(f"Number of LRR regions: {results['num_lrr_regions']}")
        
        for i, (seq, (start, end)) in enumerate(zip(results['lrr_sequences'], 
                                                   results['lrr_positions'])):
            print(f"\nLRR Region {i+1}:")
            print(f"Position: {start}-{end}")
            print(f"Sequence: {seq}")
            print(f"Length: {len(seq)}")

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
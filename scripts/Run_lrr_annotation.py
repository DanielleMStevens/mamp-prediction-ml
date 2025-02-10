from geom_lrr import Loader, Analyzer, Plotter

def main():
    # Initialize objects
    L = Loader()
    A = Analyzer()
    P = Plotter()

    # Cache settings
    pdb_cached = False
    geometry_cached = False
    regressions_cached = False
    make_cache = True

    # Load or retrieve data
    if pdb_cached:
        L.retrieve('../LRR-Annotation/cache')
    else:
        # Changed input directory to use modeled structures
        L.load_batch('../out_data/modeled_structures/pdb_for_lrr_annotator/')
        
    if geometry_cached:
        A.retrieve_geometry('../LRR-Annotation/cache')
    else:
        A.load_structures(L.structures)
        A.compute_windings()
        
    if regressions_cached:
        A.retrieve_regressions('../LRR-Annotation/cache')
    else:
        A.compute_regressions()

    # Cache data if requested
    if make_cache:
        L.cache('../LRR-Annotation/cache')
        A.cache_geometry('../LRR-Annotation/cache')
        A.cache_regressions('../LRR-Annotation/cache')

    # Load data into plotter
    P.load(A.windings, A.breakpoints, A.slopes)

    # Generate and save plots
    P.plot_regressions(save=True, directory='../out_data/lrr_annotation_plots')

if __name__ == "__main__":
    main() 
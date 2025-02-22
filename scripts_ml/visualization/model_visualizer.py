import torch
import os
import sys
from pathlib import Path
import warnings

# Try importing optional dependencies
HAVE_TENSORBOARD = False
HAVE_TORCHVIZ = False
HAVE_GRAPHVIZ = False

def print_installation_instructions():
    """Print installation instructions for missing dependencies"""
    print("\nMissing optional dependencies:")
    if not HAVE_TENSORBOARD:
        print("\ntensorboard:")
        print("  pip install tensorboard")
    if not HAVE_GRAPHVIZ:
        print("\ngraphviz:")
        print("  # First install system package:")
        print("  brew install graphviz  # macOS")
        print("  # or")
        print("  sudo apt-get install graphviz  # Ubuntu/Debian")
        print("  # Then install Python package:")
        print("  pip install graphviz")
    if not HAVE_TORCHVIZ:
        print("\ntorchviz:")
        print("  pip install torchviz")
    print("\nTo enable all visualizations, install the missing packages and run this script again.")

try:
    from torch.utils.tensorboard import SummaryWriter
    HAVE_TENSORBOARD = True
except ImportError:
    warnings.warn("tensorboard not installed. TensorBoard visualization will be disabled.")

try:
    import torchviz
    HAVE_TORCHVIZ = True
except ImportError:
    warnings.warn("torchviz not installed. Some visualizations will be disabled.")

try:
    from graphviz import Digraph
    HAVE_GRAPHVIZ = True
except ImportError:
    warnings.warn("graphviz not installed. Will use ASCII visualization instead.")

# Add the parent directory to the path to import the model
sys.path.append(str(Path(__file__).parent.parent))
from models.esm_receptor_chemical import ESMReceptorChemical

def create_ascii_visualization(save_dir='visualizations'):
    """
    Creates a simple ASCII-based visualization of the model architecture.
    This is a fallback when graphviz is not available.
    
    Args:
        save_dir (str): Directory to save the visualization
    """
    os.makedirs(save_dir, exist_ok=True)
    
    ascii_diagram = """
ESMReceptorChemical Architecture
===============================

Input Layer
----------
[Peptide Sequence]     [Receptor Sequence]
       |                      |
       v                      v
   [ESM Backbone]        [ESM Backbone]
       |                      |
       v                      v
[Sequence Features]    [Receptor Features]
       |                      |
       +----------------------+
                |
                v
    [Bidirectional Attention Layer]
         |               |
    [Sequence         [Receptor
    Bulkiness]       Bulkiness]
         |               |
         +---------------+
                |
                v
        [Feature Fusion]
                |
                v
          [Classifier]
                |
                v
        [Output (3 classes)]

Legend:
--------------------
[]     Component
|      Data flow
+      Merge/Join
v      Direction
    """
    
    with open(os.path.join(save_dir, 'model_architecture.txt'), 'w') as f:
        f.write(ascii_diagram)
    print("ASCII visualization saved as 'model_architecture.txt'")

def create_model_summary(model, save_dir='visualizations'):
    """
    Creates a detailed text summary of the model architecture.
    
    Args:
        model: The PyTorch model
        save_dir (str): Directory to save the summary
    """
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, 'model_summary.txt'), 'w') as f:
        f.write("ESMReceptorChemical Model Architecture\n")
        f.write("=" * 50 + "\n\n")
        
        # Main components
        f.write("1. ESM Backbone\n")
        f.write("   - ESM2 (t30_150M_UR50D)\n")
        f.write("   - First 20 layers frozen\n\n")
        
        f.write("2. FiLM with Attention\n")
        f.write("   - Bidirectional attention mechanism\n")
        f.write("   - Peptide and receptor projections\n")
        f.write("   - Bulkiness feature processing\n")
        f.write("   - Feature fusion layers\n\n")
        
        f.write("3. Classifier\n")
        f.write("   - Three-layer architecture\n")
        f.write("   - Layer normalization and dropout\n")
        f.write("   - Residual connections\n\n")
        
        # Parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Trainable Parameters: {trainable_params:,}\n")
        f.write(f"Frozen Parameters: {total_params - trainable_params:,}\n")
        
        # Layer-wise parameter counts
        f.write("\nLayer-wise Parameter Distribution:\n")
        f.write("-" * 30 + "\n")
        for name, module in model.named_children():
            params = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            f.write(f"{name}:\n")
            f.write(f"  Total parameters: {params:,}\n")
            f.write(f"  Trainable parameters: {trainable:,}\n")
            f.write(f"  Percentage of model: {params/total_params*100:.2f}%\n\n")

def create_tensorboard_graph(model, save_dir='visualizations'):
    """
    Creates a TensorBoard visualization if available.
    
    Args:
        model: The PyTorch model
        save_dir (str): Directory to save the visualization
    """
    if not HAVE_TENSORBOARD:
        print("Skipping TensorBoard visualization (tensorboard not installed)")
        return
        
    # Create dummy input
    batch_size = 1
    seq_length = 32
    dummy_input = {
        'peptide_x': {
            'input_ids': torch.randint(0, 100, (batch_size, seq_length)),
            'attention_mask': torch.ones(batch_size, seq_length)
        },
        'receptor_x': {
            'input_ids': torch.randint(0, 100, (batch_size, seq_length)),
            'attention_mask': torch.ones(batch_size, seq_length)
        },
        'sequence_bulkiness': torch.rand(batch_size),
        'receptor_bulkiness': torch.rand(batch_size)
    }
    
    # Create TensorBoard writer
    writer = SummaryWriter(os.path.join(save_dir, 'tensorboard'))
    try:
        writer.add_graph(model, dummy_input)
        print("TensorBoard graph created successfully")
    except Exception as e:
        print(f"Failed to create TensorBoard graph: {str(e)}")
    finally:
        writer.close()

def create_graphviz_visualization(save_dir='visualizations'):
    """
    Creates a simplified component-level visualization using graphviz.
    
    Args:
        save_dir (str): Directory to save the visualization
    """
    if not HAVE_GRAPHVIZ:
        print("Skipping Graphviz visualization (graphviz not installed)")
        return
        
    os.makedirs(save_dir, exist_ok=True)
    
    dot = Digraph(comment='ESMReceptorChemical Components')
    dot.attr(rankdir='TB')
    
    # Style configurations
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
    dot.attr('edge', color='gray50', fontcolor='gray50')
    
    # Add nodes with improved styling
    with dot.subgraph(name='cluster_0') as c:
        c.attr(label='Input Processing', style='rounded', color='blue', fontcolor='blue')
        c.attr('node', fillcolor='lightcyan')
        c.node('input_seq', 'Peptide\nSequence')
        c.node('input_rec', 'Receptor\nSequence')
        c.node('input_bulk_seq', 'Sequence\nBulkiness')
        c.node('input_bulk_rec', 'Receptor\nBulkiness')
    
    with dot.subgraph(name='cluster_1') as c:
        c.attr(label='ESM Backbone', style='rounded', color='darkgreen', fontcolor='darkgreen')
        c.attr('node', fillcolor='lightgreen')
        c.node('esm', 'ESM2 Model\n(t30_150M_UR50D)')
    
    with dot.subgraph(name='cluster_2') as c:
        c.attr(label='FiLM with Attention', style='rounded', color='darkred', fontcolor='darkred')
        c.attr('node', fillcolor='mistyrose')
        c.node('film_attention', 'Bidirectional\nAttention')
        c.node('film_fusion', 'Feature\nFusion')
    
    dot.attr('node', fillcolor='peachpuff')
    dot.node('classifier', 'Classifier\n(3 classes)')
    
    # Add edges with labels
    dot.edge('input_seq', 'esm', 'embed')
    dot.edge('input_rec', 'esm', 'embed')
    dot.edge('esm', 'film_attention', 'features')
    dot.edge('input_bulk_seq', 'film_attention', 'condition')
    dot.edge('input_bulk_rec', 'film_attention', 'condition')
    dot.edge('film_attention', 'film_fusion', 'attend')
    dot.edge('film_fusion', 'classifier', 'classify')
    
    # Save the visualization in multiple formats
    for fmt in ['png', 'pdf']:
        try:
            dot.render(os.path.join(save_dir, 'model_components'), format=fmt, cleanup=True)
            print(f"Component visualization saved in {fmt} format")
        except Exception as e:
            print(f"Failed to save {fmt} visualization: {str(e)}")

def main():
    """Main function to create all visualizations"""
    save_dir = 'visualizations'
    os.makedirs(save_dir, exist_ok=True)
    
    # Print installation instructions if any dependencies are missing
    if not all([HAVE_TENSORBOARD, HAVE_GRAPHVIZ, HAVE_TORCHVIZ]):
        print_installation_instructions()
    
    # Initialize model
    class Args:
        def __init__(self):
            pass
    args = Args()
    model = ESMReceptorChemical(args)
    
    # Create all visualizations
    print("\nCreating model visualizations...")
    create_model_summary(model, save_dir)
    print("Model summary created")
    
    create_tensorboard_graph(model, save_dir)
    
    if HAVE_GRAPHVIZ:
        create_graphviz_visualization(save_dir)
    else:
        create_ascii_visualization(save_dir)
    
    print(f"\nAll visualizations have been created in the '{save_dir}' directory")
    if HAVE_TENSORBOARD:
        print("You can view the TensorBoard visualization by running:")
        print(f"tensorboard --logdir {save_dir}/tensorboard")
    
    # Final status report
    print("\nVisualization Status:")
    print(f"✓ Text summary created: model_summary.txt")
    print(f"{'✓' if HAVE_TENSORBOARD else '✗'} TensorBoard visualization")
    print(f"{'✓' if HAVE_GRAPHVIZ else '✓'} {'Graphviz' if HAVE_GRAPHVIZ else 'ASCII'} visualization")

if __name__ == "__main__":
    main() 
from Bio.PDB import *
import os
import glob

def convert_cif_to_pdb(input_dir="./../out_data/modeled_structures", output_dir="./../out_data/pdb_structures"):
    """
    Convert all .cif files in input_dir to .pdb format and save them in output_dir
    
    Parameters
    ----------
    input_dir : str
        Directory containing .cif files
    output_dir : str
        Directory where .pdb files will be saved
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Get all .cif files in input directory
    cif_files = glob.glob(os.path.join(input_dir, "*.cif"))
    
    # Initialize parser
    parser = MMCIFParser()
    
    # Initialize PDBIO
    io = PDBIO()
    
    for cif_file in cif_files:
        try:
            # Get base filename without extension
            base_name = os.path.splitext(os.path.basename(cif_file))[0]
            
            # Parse structure from CIF file
            structure = parser.get_structure(base_name, cif_file)
            
            # Prepare output filename
            pdb_file = os.path.join(output_dir, f"{base_name}.pdb")
            
            # Set the structure to save
            io.set_structure(structure)
            
            # Save as PDB file
            io.save(pdb_file)
            
            print(f"Successfully converted {cif_file} to {pdb_file}")
            
        except Exception as e:
            print(f"Error converting {cif_file}: {str(e)}")

if __name__ == "__main__":
    convert_cif_to_pdb() 
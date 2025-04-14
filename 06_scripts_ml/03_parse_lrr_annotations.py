#-----------------------------------------------------------------------------------------------
# Krasileva Lab - Plant & Microbial Biology Department UC Berkeley
# Author: Danielle M. Stevens
# Last Updated: 07/06/2020
# Script Purpose: 
# Inputs: 
# Outputs: 
#-----------------------------------------------------------------------------------------------

def parse_lrr_annotation_results(file_path):
    """Parse the LRR annotation results file to extract domain information."""
    lrr_domains = {}
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        if line.startswith('|') or not line.strip():
            continue
            
        parts = line.strip().split('\t')
        if len(parts) >= 8:
            pdb_filename = parts[0]
            start_pos = parts[2]
            end_pos = parts[3]
            sequence = parts[7]
            
            # Extract the protein name from the PDB filename
            protein_name = pdb_filename.replace('.pdb', '')
            
            # Store the LRR domain information
            if protein_name not in lrr_domains:
                lrr_domains[protein_name] = {
                    'start': start_pos,
                    'end': end_pos,
                    'sequence': sequence
                }
    
    return lrr_domains

def parse_full_length_fasta(file_path):
    """Parse the full-length receptor FASTA file to get protein information."""
    protein_info = {}
    current_header = None
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Extract protein name and ID from the header
                current_header = line[1:]  # Remove '>'
                parts = current_header.split('|')
                if len(parts) >= 2:
                    species = parts[0].strip()
                    protein_id = parts[1].strip()
                    protein_type = parts[2].strip() if len(parts) > 2 else ""
                    
                    # Create a key that matches the format in lrr_domains
                    key_parts = [word for word in species.split() + [protein_id]]
                    key = "_".join(key_parts)
                    
                    protein_info[key] = {
                        'header': current_header,
                        'species': species,
                        'protein_id': protein_id,
                        'protein_type': protein_type
                    }
    
    return protein_info

def create_lrr_domain_fasta(lrr_domains, protein_info, output_file):
    """Create a FASTA file with LRR domain sequences."""
    with open(output_file, 'w') as f:
        for protein_name, domain_data in lrr_domains.items():
            # Skip placeholder entries
            if protein_name == "PDB_Filename" or domain_data['sequence'] == "Sequence":
                continue
                
            # Try to find matching protein info
            matching_info = None
            for key, info in protein_info.items():
                if protein_name.startswith(key) or key.startswith(protein_name):
                    matching_info = info
                    break
            
            # If no exact match, try a more flexible approach
            if not matching_info:
                for key, info in protein_info.items():
                    protein_id = info['protein_id']
                    if protein_id in protein_name:
                        matching_info = info
                        break
            
            # Create header
            if matching_info:
                header = f">{matching_info['species']}|{matching_info['protein_id']}|{matching_info['protein_type']}|LRR_domain"
            else:
                header = f">{protein_name}|LRR_domain"
            
            # Write to FASTA file
            f.write(f"{header}\n")
            f.write(f"{domain_data['sequence']}\n")

def main():
    # File paths
    lrr_annotation_file = "03_out_data/lrr_annotation_results.txt"
    full_length_fasta = "03_out_data/receptor_full_length.fasta"
    output_fasta = "03_out_data/lrr_domain_sequences.fasta"
    
    # Parse input files
    lrr_domains = parse_lrr_annotation_results(lrr_annotation_file)
    protein_info = parse_full_length_fasta(full_length_fasta)
    
    # Create output FASTA file
    create_lrr_domain_fasta(lrr_domains, protein_info, output_fasta)
    
    print(f"Created LRR domain FASTA file with {len(lrr_domains)} entries.")

if __name__ == "__main__":
    main()
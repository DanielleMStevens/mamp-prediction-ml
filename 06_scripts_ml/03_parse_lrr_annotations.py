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
    current_sequence = []
    current_header = None
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Process previous sequence if it exists
                if current_sequence and current_header:
                    protein_info[current_header]['sequence'] = ''.join(current_sequence)
                    current_sequence = []
                
                # Extract protein name and ID from the header
                current_header = line[1:]  # Remove '>'
                parts = current_header.split('|')
                if len(parts) >= 3:
                    species = parts[0].strip()
                    protein_id = parts[1].strip()
                    protein_type = parts[2].strip()
                    
                    protein_info[current_header] = {
                        'species': species,
                        'protein_id': protein_id,
                        'protein_type': protein_type,
                        'original_header': current_header
                    }
            else:
                current_sequence.append(line)
        
        # Process the last sequence
        if current_sequence and current_header:
            protein_info[current_header]['sequence'] = ''.join(current_sequence)
    
    return protein_info

def create_lrr_domain_fasta(lrr_domains, protein_info, output_file):
    """Create a FASTA file with LRR domain sequences."""
    with open(output_file, 'w') as f:
        for protein_name, domain_data in lrr_domains.items():
            # Skip placeholder entries
            if protein_name == "PDB_Filename" or domain_data['sequence'] == "Sequence":
                continue
            
            # Convert the protein name from alphafold format to desired format
            # Examples: 
            # Arabidopsis_thaliana_AT5G46330_RD_FLS2 -> Arabidopsis thaliana|AT5G46330_RD|FLS2
            # Arabidopsis_thaliana_AT5G46330_E321L_FLS2 -> Arabidopsis thaliana|AT5G46330_E321L|FLS2
            # Solanum_lycopersicum_Solyc02g070890_FLS2 -> Solanum lycopersicum|Solyc02g070890|FLS2
            parts = protein_name.split('_')
            
            # Find where the species name ends and the ID begins
            species_parts = []
            remaining_parts = []
            
            # Common ID prefixes and patterns
            id_patterns = [
                'AT', 'HE', 'XP', 'Solyc', 'Vigun', 'PGSC', 'UTN',
                'Bra', 'Glyma', 'Phvul', 'Vradi', 'Lp', 'OQ', 'PQ',
                'MH', 'AC', 'BK'
            ]
            
            # Handle special cases where species name contains numbers
            if any(p in protein_name for p in ['Group1', 'Group2', 'Group3']):
                group_idx = next(i for i, p in enumerate(parts) if 'Group' in p)
                species_parts = parts[:group_idx + 1]
                remaining_parts = parts[group_idx + 1:]
            else:
                # Standard processing
                for i, part in enumerate(parts):
                    # Check if this part starts with any ID pattern
                    # or matches accession number pattern (letters, numbers, dots)
                    # or matches Solyc pattern (e.g., Solyc02g070890)
                    if (any(part.startswith(prefix) for prefix in id_patterns) or
                        (part[0].isalpha() and any(c.isdigit() for c in part)) or
                        (part.startswith('Solyc') and 'g' in part)):
                        remaining_parts = parts[i:]
                        break
                    species_parts.append(part)
            
            # If no clear split found, use default splitting
            if not remaining_parts and species_parts:
                species_parts = parts[:2]
                remaining_parts = parts[2:]
            
            species = ' '.join(species_parts)
            
            # Handle the remaining parts to extract protein ID and type
            if not remaining_parts:
                # Fallback if parsing failed
                protein_id = protein_name
                protein_type = "UNKNOWN"
            else:
                # Get the base ID (first part)
                protein_id = remaining_parts[0]
                
                # Check for variant/mutation identifiers
                if len(remaining_parts) > 2:
                    second_part = remaining_parts[1]
                    # Variant patterns: RD, DD, E321L, etc.
                    if (second_part.isupper() or  # RD, DD
                        (second_part[0].isupper() and any(c.isdigit() for c in second_part)) or  # E321L
                        second_part.startswith(('Borsk', 'Ciste', 'Ct', 'Dobra', 'Gu', 'Mammo', 'Petro', 'Rovero', 'Shigu', 'Yeg'))):  # Natural variants
                        protein_id += f"_{second_part}"
                        protein_type = remaining_parts[2]
                    else:
                        protein_type = second_part
                else:
                    protein_type = remaining_parts[-1]
                
                # Clean up protein type if it contains additional identifiers
                if '|' in protein_type:
                    protein_type = protein_type.split('|')[0]
            
            # Construct the header in the desired format
            header = f">{species}|{protein_id}|{protein_type}|LRR_domain"
            
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
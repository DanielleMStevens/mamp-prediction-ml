#!/usr/bin/env python3  

def read_receptor_sequences(fasta_file):
    """Read the receptor FASTA file and create a mapping of sequences to headers"""
    header_map = {}
    current_header = ""
    current_sequence = ""
    
    with open(fasta_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Store previous sequence if it exists
                if current_sequence:
                    header_map[current_sequence] = current_header
                current_header = line
                current_sequence = ""
            else:
                current_sequence += line
        
        # Store the last sequence
        if current_sequence:
            header_map[current_sequence] = current_header
    
    return header_map

def find_best_match(lrr_sequence, full_sequences):
    """Find the full sequence that exactly matches the LRR sequence"""
    for full_seq, header in full_sequences.items():
        # Count matching residues in a sliding window
        lrr_len = len(lrr_sequence)
        
        for i in range(len(full_seq) - lrr_len + 1):
            window = full_seq[i:i+lrr_len]
            # Check for exact match
            if window == lrr_sequence:
                return header
    
    return None

def parse_lrr_results(lrr_file, sequence_map):
    """Parse LRR annotation results and match with full sequences"""
    sequences = []
    
    with open(lrr_file) as f:
        # Skip header line
        next(f)
        
        for line in f:
            fields = line.strip().split('\t')
            lrr_sequence = fields[7]
            
            # Find best matching full sequence
            full_header = find_best_match(lrr_sequence, sequence_map)
            
            if full_header:
                sequences.append((full_header, lrr_sequence))
            else:
                print(f"Warning: No matching sequence found for LRR: {lrr_sequence[:50]}...")
    
    return sequences

def write_fasta(sequences, output_file):
    """Write sequences in FASTA format"""
    with open(output_file, 'w') as f:
        for header, sequence in sequences:
            # Add |LRR_domain to the header
            header = header + "|LRR_domain"
            f.write(f"{header}\n")
            f.write(f"{sequence}\n")

def create_combined_fasta(sequences, excel_file, output_file):
    """Create FASTA file with LRR and ligand sequences
    
    Args:
        sequences: List of tuples (header, lrr_sequence) from LRR parsing
        excel_file: Path to Excel file containing receptor and ligand information
        output_file: Output FASTA file path
    """
    import pandas as pd
    
    # Read the Excel file
    df = pd.read_excel(excel_file)
    
    # Print column names to verify we have the correct columns
    print("Excel file columns:", df.columns.tolist())
    
    with open(output_file, 'w') as f:
        # Process each row in the Excel file
        for _, row in df.iterrows():
            try:
                # Extract required fields, using consistent column names
                species = str(row['Plant species']).strip()
                receptor = str(row['Receptor']).strip()
                genbank_id = str(row['Locus ID/Genbank']).strip()
                ligand_name = str(row['Ligand']).strip()
                ligand_seq = str(row['Ligand Sequence']).strip() if pd.notna(row['Ligand Sequence']) else ''
                
                # Find matching LRR sequence from our parsed sequences
                matching_lrr = None
                for header, lrr_seq in sequences:
                    if genbank_id in header:
                        matching_lrr = lrr_seq
                        break
                
                if matching_lrr is None:
                    print(f"Warning: No matching LRR sequence found for {genbank_id}")
                    continue
                
                # Format the header and sequence
                header = f">{species}|{receptor}|{genbank_id}:{ligand_name}"
                combined_seq = f"{matching_lrr}:{ligand_seq}"
                
                # Write to FASTA file
                f.write(f"{header}\n")
                f.write(f"{combined_seq}\n")
                
            except KeyError as e:
                print(f"Error: Missing required column in Excel file: {e}")
            except Exception as e:
                print(f"Error processing row: {e}")
                continue


def main():
    # Input files
    receptor_fasta = "03_out_data/receptor_full_length.fasta"
    lrr_results = "03_out_data/lrr_annotation_results.txt"
    excel_file = "02_in_data/All_LRR_PRR_ligand_data.xlsx"
    
    # Output files
    output_fasta = "03_out_data/lrr_domain_sequences.fasta"
    combined_output = "03_out_data/lrr_ligand_combined.fasta"
    
    # Read receptor sequences
    sequence_map = read_receptor_sequences(receptor_fasta)
    
    # Parse LRR results and match with sequences
    sequences = parse_lrr_results(lrr_results, sequence_map)
    
    # Write output FASTA file
    write_fasta(sequences, output_fasta)
    
    # Create combined FASTA file with LRR and ligand sequences
    create_combined_fasta(sequences, excel_file, combined_output)
    
    print(f"Created {output_fasta} with {len(sequences)} sequences")
    print(f"Created {combined_output} with LRR and ligand sequences")

if __name__ == "__main__":
    main() 
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
    """Find the full sequence that best matches the LRR sequence"""
    best_match = None
    best_match_count = 0
    
    for full_seq, header in full_sequences.items():
        # Count matching residues in a sliding window
        lrr_len = len(lrr_sequence)
        best_window_match = 0
        
        for i in range(len(full_seq) - lrr_len + 1):
            window = full_seq[i:i+lrr_len]
            match_count = sum(1 for a, b in zip(window, lrr_sequence) if a == b)
            best_window_match = max(best_window_match, match_count)
        
        # If this sequence has more matches than previous best, update best match
        if best_window_match > best_match_count:
            best_match_count = best_window_match
            best_match = header
            
        # If we have a near-perfect match, we can stop searching
        if best_match_count >= len(lrr_sequence) * 0.95:
            break
    
    # Only return a match if it's good enough (e.g., 90% identity)
    if best_match_count >= len(lrr_sequence) * 0.9:
        return best_match
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

def main():
    # Input files
    receptor_fasta = "out_data/receptor_full_length.fasta"
    lrr_results = "out_data/lrr_annotation_results.txt"
    
    # Output file
    output_fasta = "out_data/lrr_domain_sequences.fasta"
    
    # Read receptor sequences
    sequence_map = read_receptor_sequences(receptor_fasta)
    
    # Parse LRR results and match with sequences
    sequences = parse_lrr_results(lrr_results, sequence_map)
    
    # Write output FASTA file
    write_fasta(sequences, output_fasta)
    
    print(f"Created {output_fasta} with {len(sequences)} sequences")

if __name__ == "__main__":
    main() 
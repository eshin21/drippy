def replace_fasta_sequences(fas_path, seq_path, output_path):
    """
    Extracts headers from a .fas file and pairs them with sequences from a text file.
    """
    headers = []
    
    # Extract headers from the original FASTA file
    with open(fas_path, 'r') as fas_file:
        for line in fas_file:
            line = line.strip()
            if line.startswith(">"):
                headers.append(line)
                
    # Extract the replacement sequences, ignoring empty lines
    with open(seq_path, 'r') as seq_file:
        sequences = [line.strip() for line in seq_file if line.strip()]
        
    # Validate that the counts match to prevent misalignment
    if len(headers) != len(sequences):
        raise ValueError(f"Length mismatch: {len(headers)} headers found, but {len(sequences)} sequences provided.")
        
    # Write the new paired data to the output file
    with open(output_path, 'w') as out_file:
        for header, seq in zip(headers, sequences):
            out_file.write(f"{header}\n{seq}\n")

# Example usage:
# replace_fasta_sequences('original.fas', 'new_sequences.txt', 'repaired_output.fas')
# %%


if __name__ == "__main__":

# P0A153 - done 
# P0DN68 -- done
# P37452 -- done
# Q87KN2 -- done 
# Q92PW3 -- done
# P0CAW8 -- 2x, done 
# CollecTF_FASTA/FNR_CRP/Vibrio_vulnificus_YJ016/TF_CRP_Q7M7I9.fas
# CollecTF_FASTA/LexA/Escherichia_coli_str__K-12_substr__MG1655/TF_LexA_P0A7C2.fas'

# CollecTF_FASTA/OmpR/Yersinia_pseudotuberculosis_YPIII/TF_OmpR_Q7CFX0.fas
    fas = 'CollecTF_FASTA/OmpR/Yersinia_pseudotuberculosis_YPIII/TF_OmpR_Q7CFX0.fas'


    txt = 'CollecTF_FASTA/OmpR/Yersinia_pseudotuberculosis_YPIII/TF_OmpR_Q7CFX0.txt'

    replace_fasta_sequences(fas, txt, 'fixed.fas')
### generating report of given UniProt IDs from Ivan 

import drippy as dp
import os
import pandas as pd
from types import SimpleNamespace


# %%
def import_txt(filepath):
    out_records = []
    error_records = []
    
    current_pattern = None
    current_family = None
    
    # Mapping for families where folder name differs from family name
    folder_mapping = {
        'LexA': 'LexA',
        'CRP': 'FNR_CRP',
        'OmpR': 'OmpR'
    }
    
    base_fasta_dir = "CollecTF_FASTA"

    #import = readlines("May_Known_motifs.txt")

    #out_df = columns('ProspectivePattern', 'Family', 'UniProtID', 'Filepath', 'note')

    # error_df: same structure as out_df but to store missing ones
            
    for i in import:
        Family = 'tmp' # make this accessible to all if statements to map to the appropriate folder 

        if line starts with DR or IR 
            ProspectivePattern = DR or IR
            Family = text after "-"
        
        if line != starts with DR or IR:
            UniProtID = # uniprot ID is every line that has a continuous text string, no hyphens or spaces , save only the continuous chunk of text for each line

            # Note = save anything on the line that isn't part of the UniProt ID, e.g. P0CAW8 ( Caulobacter crescentus NA1000) save "Caulobacter crescentus NA1000" as a note

            Filepath cases: 
            if Family = 'LexA':
                SearchFolder = 'CollecTF_FASTA/LexA/'

                filepath =
                 'CollecTF_FASTA/{family}/{species can vary, search all}/TF_LexA_{UniProtID}.fas'
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
                check if file exists, if fails store in error_df with the filepath/folder you tried.
            
            elif Family = 'CRP':
                SearchFolder =  'CollecTF_FASTA/FNR_CRP/'
                ... 
            elif Family = 'OmpR':
                SearchFolder =  'CollecTF_FASTA/OmpR/'
                ... 
            if line.startswith("IR -") or line.startswith("DR -"):
                parts = line.split("-", 1)
                current_pattern = parts[0].strip()
                current_family = parts[1].strip()
            else:
                # Split once to separate the UniProtID from the rest of the line's note text
                parts = line.split(maxsplit=1)
                uniprot_id = parts[0]
                note = parts[1] if len(parts) > 1 else ""
                
            
    return SimpleNameSpace(out_df, error_df)
                # Determine folder to search in
                search_folder = folder_mapping.get(current_family, current_family)
                search_dir = os.path.join(base_fasta_dir, search_folder)
                
                found_path = None
                
                if os.path.exists(search_dir):
                    # Search for the .fas file recursively
                    for root, dirs, files in os.walk(search_dir):
                        for file in files:
                            if uniprot_id in file and file.endswith('.fas'):
                                found_path = os.path.join(root, file)
                                break
                        if found_path:
                            break
                            
                record = {
                    'ProspectivePattern': current_pattern,
                    'Family': current_family,
                    'UniProtID': uniprot_id,
                    'Filepath': found_path,
                    'Note': note
                }
                
                if found_path:
                    out_records.append(record)
                else:
                    error_records.append(record)
                    
    out_df = pd.DataFrame(out_records, columns=['ProspectivePattern', 'Family', 'UniProtID', 'Filepath', 'Note'])
    error_df = pd.DataFrame(error_records, columns=['ProspectivePattern', 'Family', 'UniProtID', 'Filepath', 'Note'])
    
    return SimpleNamespace(out_df=out_df, error_df=error_df)
        





# %%
# 1. Setup folders
report_dir = "OUTPUT_REPORT"
img_dir = os.path.join(report_dir, "images")
os.makedirs(img_dir, exist_ok=True)




# %% 

# 2. Start the Markdown document
markdown_lines = []
markdown_lines.append("# Motif Analysis Report\n")
markdown_lines.append("| UniProt ID | Direction | WebLogo | Top Candidates & P-val | Plots (Matrix & Bootstrap) |")
markdown_lines.append("|---|---|---|---|---|")

# Your list of UniProt IDs


uniprot_ids = ["Q8PN77", "C1F978"]



for uid in uniprot_ids:
    # (Find the filepath for this 'uid' using your split_fasta_by_organism logic)
    filepath = find_fasta_for_uniprot(uid) 
    
    # Analyze Direct and Reverse
    for direction in ['direct', 'reverse']:
        
        # Run your existing function
        res = detect_patterns(
            import_filepath=filepath,
            export_filepath=f"OUTPUT/{direction}_{uid}",
            direction=direction,
            metric='PIC-JSD'
        )
        
        # 3. Save Images
        logo_path = f"images/{uid}_weblogo.png"
        matrix_path = f"images/{uid}_{direction}_matrix.png"
        boot_path = f"images/{uid}_{direction}_boot.png"
        
        # Save Weblogo (Only need to do this once per UniProt ID, but doing it here is fine)
        res.motif.weblogo(os.path.join(report_dir, logo_path), format='png')
        
        # Save Matplotlib figures
        res.plots['matrix'].savefig(os.path.join(report_dir, matrix_path), bbox_inches='tight')
        res.plots['bootstrap_histogram'].savefig(os.path.join(report_dir, boot_path), bbox_inches='tight')
        
        # 4. Format the text data
        # Grab top 3 candidates and format them as an HTML/Markdown string so they fit in the table cell
        top_candidates = res.mapped_result.head(3)[['score', 'group1', 'group2']]
        candidates_html = top_candidates.to_html(index=False).replace('\n', '')
        
        # Format the table row
        row = (
            f"| **{uid}** "
            f"| {direction.capitalize()} "
            f"| ![{uid} Logo]({logo_path}) "
            f"| **p={res.pval:.4e}**<br><br>{candidates_html} "
            f"| ![{uid} Matrix]({matrix_path})<br>![{uid} Boot]({boot_path}) |"
        )
        markdown_lines.append(row)

# 5. Write the final report
with open(os.path.join(report_dir, "report.md"), "w") as f:
    f.write("\n".join(markdown_lines))

print("Report generated successfully!")

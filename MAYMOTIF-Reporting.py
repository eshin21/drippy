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

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("IR -") or line.startswith("DR -"):
                parts = line.split("-", 1)
                current_pattern = parts[0].strip()
                current_family = parts[1].strip()
            else:
                # Split once to separate the UniProtID from the rest of the line's note text
                parts = line.split(maxsplit=1)
                uniprot_id = parts[0]
                note = parts[1] if len(parts) > 1 else ""
                
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
# 0. Create filepath directory

res = import_txt('May_Known_motifs.txt')


outdf = res.out_df


filepaths_dedupe = outdf.groupby(['ProspectivePattern', 'Family', 'UniProtID', 'Filepath']).agg({'Note': 'max'}).reset_index()

# now we have a dataframe with columns 

# ProspectivePattern Family UniProtID Filepath Note

# %%
# 1.
#  Setup output folders
report_dir = "OUTPUT_REPORT"
img_dir = os.path.join(report_dir, "images")
os.makedirs(img_dir, exist_ok=True)


# %% 

# 2. Start the Markdown document
markdown_lines = []
markdown_lines.append("# Motif Analysis Report\n")
markdown_lines.append("| UniProt ID | Analyzed Direction | WebLogo | Top Candidates & P-val | Plots (Matrix & Bootstrap) | Note | Analysis Note |")
markdown_lines.append("|---|---|---|---|---|---|---|")


# 3. Loop through UniProtID

for row in filepaths_dedupe.itertuples(index=False):

    uid = row.UniProtID
    filepath = row.Filepath
    note = row.Note

    # check if exists
    if not os.path.exists(filepath):
        print(f"Error: filepath {filepath} does not exist")
        continue
    
    # Analyze Direct and Reverse
    for direction in ['direct', 'reverse']:
        
        print(f'****[REPORTING - {filepath}]')

        # Run your existing function
        res = dp.detect_patterns(
            import_filepath=filepath,
            export_filepath=f"OUTPUT/{direction}_{uid}",
            direction=direction,
            metric='PIC-JSD',
            threshold_percentile = 80, 
            plot_title = dp.filename_to_title(filepath)

        )

                
        # 3. Save Images
        logo_path = f"images/{uid}_weblogo.png"
        matrix_path = f"images/{uid}_{direction}_matrix.png"
        boot_path = f"images/{uid}_{direction}_boot.png"
        
        # Save Weblogo (Only need to do this once per UniProt ID, but doing it here is fine)
        res.motif.weblogo(os.path.join(report_dir, logo_path), format='png')
        
        # Save Matplotlib figures (safely handle None)
        matrix_img = "No Matrix"
        if res.plots.get('matrix') is not None:
            res.plots['matrix'].savefig(os.path.join(report_dir, matrix_path), bbox_inches='tight')
            matrix_img = f"![{uid} Matrix]({matrix_path})"

        boot_fig = res.plots.get('bootstrap')
        boot_img = "No Bootstrap"
        if boot_fig is not None:
            boot_fig.savefig(os.path.join(report_dir, boot_path), bbox_inches='tight')
            boot_img = f"![{uid} Boot]({boot_path})"
        
        # 4. Format the text data
        if res.mapped_result is not None:
            all_candidates = res.mapped_result[['score', 'p_value', 'group1', 'group2']].copy()
            all_candidates['p_value'] = all_candidates['p_value'].apply(lambda x: f"{x:.4e}")
            candidates_html = all_candidates.to_html(index=False).replace('\n', '')
        else:
            candidates_html = "No candidates found"
            
        # Get analysis note if it exists
        analysis_note = getattr(res, 'threshold_note', None) or ""
        
        # Format the table row
        row = (
            f"| **{uid}** "
            f"| {direction.capitalize()} "
            f"| ![{uid} Logo]({logo_path}) "
            f"| {candidates_html} "
            f"| {matrix_img}<br>{boot_img} "
            f"| {note} "
            f"| {analysis_note} |"
        )
        markdown_lines.append(row)

# %%

# 5. Write the final report
with open(os.path.join(report_dir, "report.md"), "w") as f:
    f.write("\n".join(markdown_lines))

print("Report generated successfully!")

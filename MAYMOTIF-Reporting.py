### generating report of given UniProt IDs from Ivan 

import drippy as dp
import importlib
importlib.reload(dp)
import os
import pandas as pd
from types import SimpleNamespace
import urllib.error
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import shutil

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
                
                found_paths = []
                
                if os.path.exists(search_dir):
                    # Search for the .fas file recursively
                    for root, dirs, files in os.walk(search_dir):
                        # Skip any directory named 'OLD' so we don't grab misaligned files
                        if 'OLD' in dirs:
                            dirs.remove('OLD')
                            
                        for file in files:
                            if uniprot_id in file and file.endswith('.fas'):
                                found_paths.append(os.path.join(root, file))
                            
                if found_paths:
                    for found_path in found_paths:
                        record = {
                            'ProspectivePattern': current_pattern,
                            'Family': current_family,
                            'UniProtID': uniprot_id,
                            'Filepath': found_path,
                            'Note': note
                        }
                        out_records.append(record)
                else:
                    record = {
                        'ProspectivePattern': current_pattern,
                        'Family': current_family,
                        'UniProtID': uniprot_id,
                        'Filepath': None,
                        'Note': note
                    }
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
# 1. Setup output folders and threshold
THRESHOLD = 80
report_dir = f"{THRESHOLD}_Threshold_OUTPUT_REPORT"
out_dir = f"{THRESHOLD}_Threshold_OUTPUT"
img_dir = os.path.join(report_dir, "images")
os.makedirs(img_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)


# %% 
# --- Pre-calculate Species Protein for the dropdowns ---
def get_clean_species(filepath):
    sp = dp.filename_to_title(filepath)[1]
    return "_".join(sp) if isinstance(sp, list) else str(sp)

filepaths_dedupe['SpeciesProtein'] = filepaths_dedupe['Filepath'].apply(get_clean_species)

# Sort the dataframe so families and species are grouped together logically
filepaths_dedupe = filepaths_dedupe.sort_values(by=['Family', 'UniProtID'])

# Export clean relationship data to JSON for JavaScript cascading logic
combo_data = filepaths_dedupe[['Family', 'SpeciesProtein', 'UniProtID']].dropna().drop_duplicates().to_dict(orient='records')
combo_json = json.dumps(combo_data)


# Extract exact, clean unique values for the dropdowns
unique_families = sorted(filepaths_dedupe['Family'].dropna().unique())
unique_species = sorted(filepaths_dedupe['SpeciesProtein'].dropna().unique())
unique_uids = sorted(filepaths_dedupe['UniProtID'].dropna().unique())

# Build the HTML <option> tags
fam_opts = "".join([f"<option value='{f}'>{f}</option>" for f in unique_families])
spec_opts = "".join([f"<option value='{s}'>{s}</option>" for s in unique_species])
uid_opts = "".join([f"<option value='{u}'>{u}</option>" for u in unique_uids])

# 2. Start the HTML document with CSS and Filter Bar
html_lines = [
    "<!DOCTYPE html>",
    "<html>",
    "<head>",
    "<style>",
    "  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 20px; background-color: #f4f4f9; color: #333; }",
    "  h1 { color: #222; }",
    "  .filter-bar { display: flex; gap: 20px; margin-bottom: 20px; padding: 15px; background: #fff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }",
    "  .filter-group { display: flex; flex-direction: column; }",
    "  select { padding: 8px; border: 1px solid #ccc; border-radius: 4px; min-width: 150px; }",
    "  label { font-weight: bold; margin-bottom: 5px; font-size: 14px; }",
    "  table { border-collapse: collapse; width: 100%; background-color: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }",
    "  th, td { border: 1px solid #ddd; padding: 12px; text-align: left; vertical-align: top; }",
    "  th { background-color: #2c3e50; color: white; position: sticky; top: 0; z-index: 10; }",
    "  .weblogo { max-width: 250px; height: auto; }",
    "  .plot-img { max-width: 350px; height: auto; margin-bottom: 10px; display: block; }",
    "  .candidates-table table { font-size: 0.9em; width: 100%; border: none; box-shadow: none; }",
    "  .candidates-table th { background-color: #f8f9fa; color: #333; }",
    "  details { cursor: pointer; color: #0066cc; }",
    "  summary { font-weight: bold; margin-bottom: 10px; }",
    "</style>",
    "</head>",
    "<body>",
    "<h1>Motif Analysis Report</h1>",
    
    # Filter Bar HTML (Dynamically populated by JS logic below)
    "<div class='filter-bar'>",
    "  <div class='filter-group'><label>Search</label><input type='text' id='globalSearch' onkeyup='handleFilterChange()' placeholder='Type to search...'></div>",
    "  <div class='filter-group'><label>Family</label><select id='filterFamily' onchange='handleFilterChange()'><option value='ALL'>All</option></select></div>",
    "  <div class='filter-group'><label>Species Protein</label><select id='filterSpecies' onchange='handleFilterChange()'><option value='ALL'>All</option></select></div>",
    "  <div class='filter-group'><label>UniProt ID</label><select id='filterUniProt' onchange='handleFilterChange()'><option value='ALL'>All</option></select></div>",
    "</div>",
    
    "<table id='reportTable'>",
        "<thead>",
        "<tr><th>Family</th><th>Species Protein</th><th>UniProt ID</th><th>Prospective Pattern</th><th>Analyzed Direction</th><th>Num Sequences</th><th>WebLogo</th><th>Top Candidates & P-val</th><th>Plots (Matrix, Hist, Boot)</th><th>Note</th><th>Analysis Note</th></tr>",
        "</thead>",
        "<tbody>"
]

# Initialize empty list to store data for our final Pandas DataFrame
report_data = []


# Toggle to re-run ONLY files that have an 'OLD' folder and clear their cached outputs
RERUN_FIXED_ONLY = False

# 3. Loop through UniProtID
for row in filepaths_dedupe.itertuples(index=False):


    family = row.Family
    uid = row.UniProtID
    filepath = row.Filepath
    note = row.Note
    species_protein = row.SpeciesProtein
    prospective_pattern = row.ProspectivePattern
    
    # Check if exists
    if not os.path.exists(filepath):
        print(f"Error: filepath {filepath} does not exist")
        continue
        
    # Count the number of sequences in the .fas file (lines not starting with '>')
    seq_count = 0
    with open(filepath, 'r') as fas_f:
        for line in fas_f:
            if line.strip() and not line.startswith('>'):
                seq_count += 1

    # If toggle is on, check if an 'OLD' directory exists next to the .fas file
    if RERUN_FIXED_ONLY:
        old_folder_path = os.path.join(os.path.dirname(filepath), 'OLD')
        if not os.path.exists(old_folder_path):
            continue  # Skip files that weren't fixed
            
        # Clean up existing outputs so they are forced to regenerate
        for direction in ['direct', 'reverse']:
            files_to_remove = [
                f"{out_dir}/{direction}_{uid}_{species_protein}.xlsx",
                os.path.join(report_dir, f"images/{uid}_{species_protein}_weblogo.png"),
                os.path.join(report_dir, f"images/{uid}_{species_protein}_{direction}_matrix.png"),
                os.path.join(report_dir, f"images/{uid}_{species_protein}_{direction}_histogram.png"),
                os.path.join(report_dir, f"images/{uid}_{species_protein}_{direction}_boot.png")
            ]
            for f_path in files_to_remove:
                if os.path.exists(f_path):
                    os.remove(f_path)
                    
    for direction in ['direct', 'reverse']:
        print(f'****[REPORTING - {filepath}] - {direction}')

        res = dp.detect_patterns(
            import_filepath=filepath,
            export_filepath=f"{out_dir}/{direction}_{uid}_{species_protein}",
            direction=direction,
            metric='PIC-JSD',
            threshold_percentile=THRESHOLD, 
            min_threshold_percentile=THRESHOLD,
            plot_title=f'{uid}_{species_protein}'
        )


        
        # 3. Link Existing Images OR Generate if Missing
        logo_path = f"images/{uid}_{species_protein}_weblogo.png"
        matrix_path = f"images/{uid}_{species_protein}_{direction}_matrix.png"
        histo_path = f"images/{uid}_{species_protein}_{direction}_histogram.png"
        boot_path = f"images/{uid}_{species_protein}_{direction}_boot.png"
        
        full_logo_path = os.path.join(report_dir, logo_path)
        full_matrix_path = os.path.join(report_dir, matrix_path)
        full_histo_path = os.path.join(report_dir, histo_path)
        full_boot_path = os.path.join(report_dir, boot_path)

        # Try to copy the WebLogo from the 80% threshold report if it exists to save time
        old_logo_path = os.path.join("80_Threshold_OUTPUT_REPORT", logo_path)
        if not os.path.exists(full_logo_path) and os.path.exists(old_logo_path):
            shutil.copy(old_logo_path, full_logo_path)

        # WebLogo (with server error handling)
        logo_img = "No Logo Found"
        if os.path.exists(full_logo_path):
            logo_img = f'<img class="weblogo" src="{logo_path}" alt="{uid} {species_protein} Logo">'
        else:
            try:
                res.motif.weblogo(full_logo_path, format='png')
                logo_img = f'<img class="weblogo" src="{logo_path}" alt="{uid} {species_protein} Logo">'
            except urllib.error.HTTPError as e:
                print(f"  -> WARNING: WebLogo server down for {uid} (HTTP {e.code})")
                logo_img = "Logo Server Down"
            except Exception as e:
                print(f"  -> WARNING: Could not generate WebLogo for {uid}: {e}")

        # Matplotlib Figures
        matrix_img = "No Matrix"
        if os.path.exists(full_matrix_path):
            matrix_img = f'<img class="plot-img" src="{matrix_path}" alt="{uid} {species_protein} Matrix">'
        elif res.plots.get('matrix') is not None:
            res.plots['matrix'].savefig(full_matrix_path, bbox_inches='tight')
            matrix_img = f'<img class="plot-img" src="{matrix_path}" alt="{uid} {species_protein} Matrix">'

        histo_img = "No Histogram"
        if os.path.exists(full_histo_path):
            histo_img = f'<img class="plot-img" src="{histo_path}" alt="{uid} {species_protein} Histogram">'
        elif res.plots.get('histogram') is not None:
            res.plots['histogram'].savefig(full_histo_path, bbox_inches='tight')
            histo_img = f'<img class="plot-img" src="{histo_path}" alt="{uid} {species_protein} Histogram">'

        boot_img = "No Bootstrap"
        if os.path.exists(full_boot_path):
            boot_img = f'<img class="plot-img" src="{boot_path}" alt="{uid} {species_protein} Boot">'
        elif res.plots.get('bootstrap') is not None:
            res.plots['bootstrap'].savefig(full_boot_path, bbox_inches='tight')
            boot_img = f'<img class="plot-img" src="{boot_path}" alt="{uid} {species_protein} Boot">'
        
        # 4. Format the text data
        if res.mapped_result is not None and not res.mapped_result.empty:
            cols = ['score', 'p_value', 'group1', 'group2']
            if 'full_pattern_html' in res.mapped_result.columns:
                cols.append('full_pattern_html')
            elif 'full_pattern' in res.mapped_result.columns:
                cols.append('full_pattern')
            all_candidates = res.mapped_result[cols].copy()
            if 'full_pattern_html' in all_candidates.columns:
                all_candidates.rename(columns={'full_pattern_html': 'full_pattern'}, inplace=True)
            all_candidates['p_value'] = all_candidates['p_value'].apply(lambda x: f"{x:.4e}")
            candidates_html = all_candidates.to_html(index=False, escape=False).replace('\n', '')
        else:
            candidates_html = "No candidates found"
            
        # combine the threshold note and the bp mismatch length warning for display in the final table 

        parts = [getattr(res, 'threshold_note', ''), getattr(res, 'length_warning', '')]
        warnings_str = "<br><br>".join(p for p in parts if p)       


         # --- Append data for the subsettable DataFrame ---
        # We replace HTML line breaks with standard pipes for cleaner text
        row_dict = {
            'Family': family,
            'Species Protein': species_protein,
            'UniProt ID': uid,
            'Prospective Pattern': prospective_pattern,
            'Analyzed Direction': direction.capitalize(),
            'Num Sequences': seq_count,
            'WebLogo Path': full_logo_path if os.path.exists(full_logo_path) else None,
            'Top Candidates': res.mapped_result if res.mapped_result is not None else "No candidates found",
            'Matrix Path': full_matrix_path if os.path.exists(full_matrix_path) else None,
            'Histogram Path': full_histo_path if os.path.exists(full_histo_path) else None,
            'Bootstrap Path': full_boot_path if os.path.exists(full_boot_path) else None,
            'Note': note,
            'Analysis Note': warnings_str.replace('<br><br>', ' | ') 
        }
        report_data.append(row_dict)
        
        # Format the HTML table row
        html_row = f"""
        <tr>
            <td><strong>{family}</strong></td>
            <td><strong>{species_protein}</strong></td>
            <td><strong>{uid}</strong></td>
            <td>{prospective_pattern}</td>
            <td>{direction.capitalize()}</td>
        <td>{seq_count}</td>
            <td>{logo_img}</td>
            <td class="candidates-table">{candidates_html}</td>
            <td>
                <details>
                    <summary>View Plots</summary>
                    {matrix_img}
                    {histo_img}
                    {boot_img}
                </details>
            </td>
            <td>{note}</td>
            <td>{warnings_str}</td>
        </tr>
        """
        html_lines.append(html_row)



# 5. Append clean JavaScript for Cascading Dropdowns and Table Filtering
js_script = f"""
<script>
// Load the clean relationship data from Python
const comboData = {combo_json};

function rebuildSelect(selectId, validValues, currentValue) {{
    let select = document.getElementById(selectId);
    select.innerHTML = "<option value='ALL'>All</option>";
    
    validValues.forEach(val => {{
        let opt = document.createElement("option");
        opt.value = val;
        opt.textContent = val;
        if (val === currentValue) opt.selected = true;
        select.appendChild(opt);
    }});
    
    // If the previous selection is no longer valid under new filters, reset to ALL
    if (currentValue !== 'ALL' && !validValues.includes(currentValue)) {{
        select.value = 'ALL';
    }}
}}

function updateOptions() {{
    let selectedFam = document.getElementById("filterFamily").value;
    let selectedSpec = document.getElementById("filterSpecies").value;
    let selectedUid = document.getElementById("filterUniProt").value;
    let textSearch = document.getElementById("globalSearch").value.toUpperCase();

    // Helper to check if a row matches the text search across the three columns
    const matchesText = (row) => {{
        if (!textSearch) return true;
        const combined = (row.Family + " " + row.SpeciesProtein + " " + row.UniProtID).toUpperCase();
        return combined.includes(textSearch);
    }};

    // Filter combinations to see what is valid for EACH dropdown based on the OTHER filters
    let famRows = comboData.filter(row => 
        (selectedSpec === 'ALL' || row.SpeciesProtein === selectedSpec) && 
        (selectedUid === 'ALL' || row.UniProtID === selectedUid) &&
        matchesText(row)
    );
    let validFams = [...new Set(famRows.map(r => r.Family))].sort();
    
    let specRows = comboData.filter(row => 
        (selectedFam === 'ALL' || row.Family === selectedFam) && 
        (selectedUid === 'ALL' || row.UniProtID === selectedUid) &&
        matchesText(row)
    );
    let validSpecs = [...new Set(specRows.map(r => r.SpeciesProtein))].sort();

    let uidRows = comboData.filter(row => 
        (selectedFam === 'ALL' || row.Family === selectedFam) && 
        (selectedSpec === 'ALL' || row.SpeciesProtein === selectedSpec) &&
        matchesText(row)
    );
    let validUids = [...new Set(uidRows.map(r => r.UniProtID))].sort();

    rebuildSelect("filterFamily", validFams, selectedFam);
    rebuildSelect("filterSpecies", validSpecs, document.getElementById("filterSpecies").value); 
    rebuildSelect("filterUniProt", validUids, document.getElementById("filterUniProt").value);
}}

function filterTable() {{
    const rows = document.querySelectorAll("#reportTable > tbody > tr");
    
    const famFilter = document.getElementById("filterFamily").value.toUpperCase();
    const specFilter = document.getElementById("filterSpecies").value.toUpperCase();
    const uniFilter = document.getElementById("filterUniProt").value.toUpperCase();
    const textSearch = document.getElementById("globalSearch").value.toUpperCase();

    rows.forEach(row => {{
        const famText = row.cells[0].innerText.trim().toUpperCase();
        const specText = row.cells[1].innerText.trim().toUpperCase();
        const uniText = row.cells[2].innerText.trim().toUpperCase();

        const matchFam = (famFilter === "ALL" || famText === famFilter);
        const matchSpec = (specFilter === "ALL" || specText === specFilter);
        const matchUni = (uniFilter === "ALL" || uniText === uniFilter);
        
        // String match against the combined text of the three columns
        const combinedText = famText + " " + specText + " " + uniText;
        const matchText = (textSearch === "" || combinedText.includes(textSearch));

        if (matchFam && matchSpec && matchUni && matchText) {{
            row.style.display = "";
        }} else {{
            row.style.display = "none";
        }}
    }});
}}

// Triggered whenever a user changes any dropdown or types in the search bar
function handleFilterChange() {{
    updateOptions();
    filterTable();
}}

// Run once when the page loads to populate everything initially
window.onload = function() {{
    updateOptions();
}};
</script>
"""


# %% 
html_lines.append("</tbody></table>")
html_lines.append(js_script)
html_lines.append("</body></html>")

with open(os.path.join(report_dir, f"{THRESHOLD}_report.html"), "w") as f:
    f.write("\n".join(html_lines))

print("HTML Report generated successfully!")
# %%

# --- NEW: Generate and save the subsettable DataFrame ---

expanded_rows = []
for row in report_data:
    base_dict = {k: v for k, v in row.items() if k != 'Top Candidates'}
    candidates = row['Top Candidates']
    
    if isinstance(candidates, pd.DataFrame) and not candidates.empty:
        for _, cand_row in candidates.iterrows():
            new_row = base_dict.copy()
            new_row['coords'] = cand_row['coords']
            new_row['length'] = cand_row['length']
            new_row['score'] = cand_row['score']
            new_row['group1'] = cand_row['group1']
            new_row['group2'] = cand_row['group2']
            new_row['full_pattern'] = cand_row.get('full_pattern', None)
            new_row['p_value'] = cand_row['p_value']
            new_row['Inverted_Pval'] = 1.0 - cand_row['p_value']
            expanded_rows.append(new_row)
    else:
        new_row = base_dict.copy()
        new_row['coords'] = None
        new_row['length'] = None
        new_row['score'] = None
        new_row['group1'] = None
        new_row['group2'] = None
        new_row['full_pattern'] = None
        new_row['p_value'] = 1.0
        new_row['Inverted_Pval'] = 0.0
        expanded_rows.append(new_row)

final_report_df = pd.DataFrame(expanded_rows)

# %%



# --- Alignment with Literature ---

# 1. Create "Alignment" Column

def check_alignment(row, sig_threshold=0.05):
    # Standardize text for safe comparison
    pattern = str(row['Prospective Pattern']).strip().upper()
    direction = str(row['Analyzed Direction']).strip().upper()
    
    try:
        pval = float(row['p_value'])
    except (ValueError, TypeError):
        pval = 1.0
        
    # Check if the analyzed direction matches the literature pattern
    is_match = False
    if pattern == "IR" and direction == "REVERSE":
        is_match = True
    elif pattern == "DR" and direction == "DIRECT":
        is_match = True
        
    if is_match:
        return "Strong agree" if pval < sig_threshold else "Weak agree"
    else:
        return "Strong disagree" if pval < sig_threshold else "Weak disagree"

final_report_df['Alignment'] = final_report_df.apply(check_alignment, axis=1)

# Save the master DataFrame to CSV
final_report_df.to_excel(os.path.join(report_dir, f"{THRESHOLD}_report_data.xlsx"), index=False)
print("Master DataFrame exported to report_data.xlsx successfully!")


# 2. Extract best conclusion per observation (by min p-value)
best_idx = final_report_df.groupby(['Species Protein', 'UniProt ID'])['p_value'].idxmin()
best_conclusion_df = final_report_df.loc[best_idx].reset_index(drop=True)

best_conclusion_df.to_excel(os.path.join(report_dir, f"{THRESHOLD}_best_conclusion_data.xlsx"), index=False)
print("Best conclusion DataFrame exported to best_conclusion_data.xlsx successfully!")


# %%

# 3. Plotting ROC and PR Curves

# Pivot the full data so every observation has both a 'Direct' and 'Reverse' continuous score
pivot_df = final_report_df.pivot_table(
    index=['Species Protein', 'UniProt ID', 'Prospective Pattern'],
    columns='Analyzed Direction',
    values='Inverted_Pval',
    aggfunc='max'
).reset_index()
# %%

# Clean up pattern strings (just in case there are trailing spaces)
pivot_df['Prospective Pattern'] = pivot_df['Prospective Pattern'].astype(str).str.strip().str.upper()

# 1. Define Ground Truths (Binary 1 or 0)
y_true_dr = np.where(pivot_df['Prospective Pattern'] == 'DR', 1, 0)
y_true_ir = np.where(pivot_df['Prospective Pattern'] == 'IR', 1, 0)

# 2. Define Scores (Inverted P-values, higher is more confident)
y_score_dr = pivot_df['Direct'].fillna(0)
y_score_ir = pivot_df['Reverse'].fillna(0)

# %%


# 3. Calculate metrics for Direct Repeats
fpr_dr, tpr_dr, thresh_dr = roc_curve(y_true_dr, y_score_dr)
roc_auc_dr = auc(fpr_dr, tpr_dr)
prec_dr, recall_dr, _ = precision_recall_curve(y_true_dr, y_score_dr)
pr_auc_dr = average_precision_score(y_true_dr, y_score_dr)


# %%


# 4. Calculate metrics for Inverted Repeats
fpr_ir, tpr_ir, thresh_ir = roc_curve(y_true_ir, y_score_ir)
roc_auc_ir = auc(fpr_ir, tpr_ir)
prec_ir, recall_ir, _ = precision_recall_curve(y_true_ir, y_score_ir)
pr_auc_ir = average_precision_score(y_true_ir, y_score_ir)

# Generate Plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ROC Curves
ax1.plot(fpr_dr, tpr_dr, color='blue', lw=2, label=f'Direct Repeats (AUC = {roc_auc_dr:.2f})')
ax1.plot(fpr_ir, tpr_ir, color='red', lw=2, label=f'Inverted Repeats (AUC = {roc_auc_ir:.2f})')
ax1.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title(f'{THRESHOLD}% Threshold ROC Curves')
ax1.legend(loc='lower right')

# PR Curves
ax2.plot(recall_dr, prec_dr, color='blue', lw=2, label=f'Direct Repeats (AUC = {pr_auc_dr:.2f})')
ax2.plot(recall_ir, prec_ir, color='red', lw=2, label=f'Inverted Repeats (AUC = {pr_auc_ir:.2f})')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title(f'{THRESHOLD}% Precision-Recall Curves')
ax2.legend(loc='lower left')

plt.tight_layout()
plt.savefig(os.path.join(report_dir, f"{THRESHOLD}_ROC_PR_Curves.png"))
plt.close()
print("ROC and PR curves saved to ROC_PR_Curves.png successfully!")


# 5. Extract Optimal Thresholds and Evaluate Each Observation
# Using Youden's J statistic (TPR - FPR) to find the optimal threshold
opt_idx_dr = np.argmax(tpr_dr - fpr_dr)
opt_thresh_dr = thresh_dr[opt_idx_dr]

opt_idx_ir = np.argmax(tpr_ir - fpr_ir)
opt_thresh_ir = thresh_ir[opt_idx_ir]

def evaluate_obs(score, truth, threshold):
    if score >= threshold:
        return 'TP' if truth == 1 else 'FP'
    else:
        return 'TN' if truth == 0 else 'FN'

pivot_df[f'DR_Eval (Opt Thresh: {opt_thresh_dr:.4f})'] = [evaluate_obs(s, t, opt_thresh_dr) for s, t in zip(y_score_dr, y_true_dr)]
pivot_df[f'IR_Eval (Opt Thresh: {opt_thresh_ir:.4f})'] = [evaluate_obs(s, t, opt_thresh_ir) for s, t in zip(y_score_ir, y_true_ir)]

# Save to a new Excel file
pivot_df.to_excel(os.path.join(report_dir, f"{THRESHOLD}_observation_roc_evaluations.xlsx"), index=False)
print("Observation-level ROC evaluations exported to observation_roc_evaluations.xlsx successfully!")

# %%

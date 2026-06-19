import drippy as dp
import importlib
importlib.reload(dp)
import os
import pandas as pd
from types import SimpleNamespace
import json
import shutil
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import urllib.error

# %%
# 0. Filepath Discovery

meme_folders = [f"IMPORTS/meme_out_{i}" for i in range(1, 5)]

records = []
for folder in meme_folders:
    xml_path = os.path.join(folder, "meme.xml")
    if os.path.exists(xml_path):
        for motif_idx in range(10): # MEME motifs 0 to 9
            motif_num_display = motif_idx + 1 # Display as 1 to 10
            logo_path = os.path.join(folder, f"logo{motif_num_display}.png")
            records.append({
                'MemeFolder': os.path.basename(folder),
                'MotifIndex': motif_idx,
                'MotifNumber': str(motif_num_display),
                'Filepath': xml_path,
                'OriginalLogo': logo_path
            })

filepaths_dedupe = pd.DataFrame(records)

# Load manual annotations Excel and merge
anno_path = 'MEME manual annotations.xlsx'
if os.path.exists(anno_path):
    anno_df = pd.read_excel(anno_path)
    # Clean columns and rename
    anno_df.columns = [c.strip() for c in anno_df.columns]
    
    # Try to find the manual annotation column
    possible_cols = ['ManualAnnotation', 'Interocular Annotation', 'Annotation']
    manual_col = None
    for col in possible_cols:
        found = [c for c in anno_df.columns if c.lower() == col.lower()]
        if found:
            manual_col = found[0]
            break
            
    if manual_col is None:
        # Fallback to the third column
        manual_col = anno_df.columns[2]
        
    anno_df.rename(columns={
        'meme folder': 'MemeFolder',
        'motif': 'MotifNumber',
        manual_col: 'ManualAnnotation'
    }, inplace=True)
    
    # Standardize values to join correctly
    anno_df['MemeFolder'] = 'meme_out_' + anno_df['MemeFolder'].astype(str)
    anno_df['MotifNumber'] = anno_df['MotifNumber'].astype(str)
    anno_df['ManualAnnotation'] = anno_df['ManualAnnotation'].astype(str).str.strip().replace({'none': 'None', 'nan': 'None'})
    
    # Merge
    filepaths_dedupe = pd.merge(filepaths_dedupe, anno_df[['MemeFolder', 'MotifNumber', 'ManualAnnotation']], on=['MemeFolder', 'MotifNumber'], how='left')
    filepaths_dedupe['ManualAnnotation'] = filepaths_dedupe['ManualAnnotation'].fillna('None')
else:
    # Fallback in case file is missing
    filepaths_dedupe['ManualAnnotation'] = 'None'

# %%
# 1. Setup output folders and threshold

#########################################################
# TUNING PARAMETERS
# If a command-line argument is passed, override the thresholds for a fixed run
THRESHOLD_MIN = 80
THRESHOLD_MAX = 80
if len(sys.argv) > 1:
    try:
        val = int(sys.argv[1])
        THRESHOLD_MIN = val
        THRESHOLD_MAX = val
    except ValueError:
        # Ignore non-integer arguments passed by Jupyter/ipykernel
        pass

LABEL = THRESHOLD_MAX
min_length = 2

report_dir = f"MEME_{LABEL}_Threshold_OUTPUT_REPORT"
out_dir = f"MEME_{LABEL}_Threshold_OUTPUT"
img_dir = os.path.join(report_dir, "images")
os.makedirs(img_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)

# Export clean relationship data to JSON for JavaScript cascading logic
combo_data = filepaths_dedupe[['MemeFolder', 'MotifNumber']].dropna().drop_duplicates().to_dict(orient='records')
combo_json = json.dumps(combo_data)

# Extract exact, clean unique values for the dropdowns
unique_folders = sorted(filepaths_dedupe['MemeFolder'].dropna().unique())
unique_motifs = sorted(filepaths_dedupe['MotifNumber'].dropna().unique(), key=int)

# Build the HTML <option> tags
folder_opts = "".join([f"<option value='{f}'>{f}</option>" for f in unique_folders])
motif_opts = "".join([f"<option value='{m}'>{m}</option>" for m in unique_motifs])

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
    "<h1>MEME Motif Analysis Report</h1>",
    
    # Filter Bar HTML (Dynamically populated by JS logic below)
    "<div class='filter-bar'>",
    "  <div class='filter-group'><label>Search</label><input type='text' id='globalSearch' onkeyup='handleFilterChange()' placeholder='Type to search...'></div>",
    "  <div class='filter-group'><label>MEME Folder</label><select id='filterFolder' onchange='handleFilterChange()'><option value='ALL'>All</option></select></div>",
    "  <div class='filter-group'><label>Motif Number</label><select id='filterMotif' onchange='handleFilterChange()'><option value='ALL'>All</option></select></div>",
    "</div>",
    
    "<table id='reportTable'>",
        "<thead>",
        "<tr><th>MEME Folder</th><th>Motif Number</th><th>Manual Annotation</th><th>Analyzed Direction</th><th>MEME Logo</th><th>Top Candidates & P-val</th><th>Plots (Matrix, Hist, Boot)</th><th>Execution Time (s)</th><th>Analysis Note</th></tr>",
        "</thead>",
        "<tbody>"
]

# Initialize empty list to store data for our final Pandas DataFrame
report_data = []

# Sort the dataframe so folders and motifs are grouped together logically
filepaths_dedupe = filepaths_dedupe.sort_values(by=['MemeFolder', 'MotifIndex'])

# 3. Loop through MEME motifs
for row in filepaths_dedupe.itertuples(index=False):

    folder_name = row.MemeFolder
    motif_idx = row.MotifIndex
    motif_num = row.MotifNumber
    filepath = row.Filepath
    orig_logo = row.OriginalLogo
    manual_annotation = getattr(row, 'ManualAnnotation', 'None')
    
    # Copy original MEME logo to report images
    dest_logo_path = os.path.join(img_dir, f"{folder_name}_logo{motif_num}.png")
    if os.path.exists(orig_logo) and not os.path.exists(dest_logo_path):
        shutil.copy(orig_logo, dest_logo_path)

    logo_img = "No Logo Found"
    if os.path.exists(dest_logo_path):
        logo_img = f'<img class="weblogo" src="images/{folder_name}_logo{motif_num}.png" alt="{folder_name} Motif {motif_num} Logo">'
        
    for direction in ['direct', 'reverse']:
        print(f'****[REPORTING - {folder_name} Motif {motif_num}] - {direction}')

        plot_title = f"{folder_name}_motif{motif_num}"

        try:
            start_time = time.time()
            res = dp.detect_patterns(
                import_filepath=filepath,
                export_filepath=f"{out_dir}/{direction}_{folder_name}_motif{motif_num}",
                motif_num=motif_idx,
                direction=direction,
                metric='PIC-JSD',
                threshold_percentile=THRESHOLD_MAX, 
                min_threshold_percentile=THRESHOLD_MIN,
                min_length=min_length,
                plot_title=plot_title
            )
            exec_time = time.time() - start_time
        except Exception as e:
            print(f"  -> Error analyzing {plot_title}: {e}")
            continue

        matrix_path = f"images/{folder_name}_motif{motif_num}_{direction}_matrix.png"
        histo_path = f"images/{folder_name}_motif{motif_num}_{direction}_histogram.png"
        boot_path = f"images/{folder_name}_motif{motif_num}_{direction}_boot.png"
        
        full_matrix_path = os.path.join(report_dir, matrix_path)
        full_histo_path = os.path.join(report_dir, histo_path)
        full_boot_path = os.path.join(report_dir, boot_path)

        # Matplotlib Figures
        matrix_img = "No Matrix"
        if os.path.exists(full_matrix_path):
            matrix_img = f'<img class="plot-img" src="{matrix_path}" alt="{plot_title} Matrix">'
        elif res.plots.get('matrix') is not None:
            res.plots['matrix'].savefig(full_matrix_path, bbox_inches='tight')
            matrix_img = f'<img class="plot-img" src="{matrix_path}" alt="{plot_title} Matrix">'

        histo_img = "No Histogram"
        if os.path.exists(full_histo_path):
            histo_img = f'<img class="plot-img" src="{histo_path}" alt="{plot_title} Histogram">'
        elif res.plots.get('histogram') is not None:
            res.plots['histogram'].savefig(full_histo_path, bbox_inches='tight')
            histo_img = f'<img class="plot-img" src="{histo_path}" alt="{plot_title} Histogram">'

        boot_img = "No Bootstrap"
        if os.path.exists(full_boot_path):
            boot_img = f'<img class="plot-img" src="{boot_path}" alt="{plot_title} Boot">'
        elif res.plots.get('bootstrap') is not None:
            res.plots['bootstrap'].savefig(full_boot_path, bbox_inches='tight')
            boot_img = f'<img class="plot-img" src="{boot_path}" alt="{plot_title} Boot">'
        
        # 4. Format the text data
        if res.mapped_result is not None and not res.mapped_result.empty:
            cols = ['score', 'p_value', 'group1', 'group2']
            if 'correctness_%' in res.mapped_result.columns:
                cols.append('correctness_%')
            if 'full_pattern_html' in res.mapped_result.columns:
                cols.append('full_pattern_html')
            elif 'full_pattern' in res.mapped_result.columns:
                cols.append('full_pattern')
            if 'evaluation' in res.mapped_result.columns:
                cols.append('evaluation')
            all_candidates = res.mapped_result[cols].copy()
            if 'full_pattern_html' in all_candidates.columns:
                all_candidates.rename(columns={'full_pattern_html': 'full_pattern'}, inplace=True)
            if pd.api.types.is_numeric_dtype(all_candidates['p_value']):
                all_candidates['p_value'] = all_candidates['p_value'].apply(lambda x: f"{x:.4e}")
            candidates_html = all_candidates.to_html(index=False, escape=False).replace('\n', '')
        else:
            candidates_html = "No candidates found"
            
        parts = [getattr(res, 'threshold_note', ''), getattr(res, 'length_warning', '')]
        warnings_str = "<br><br>".join(p for p in parts if p)       

         # --- Append data for the subsettable DataFrame ---
        row_dict = {
            'MEME Folder': folder_name,
            'Motif Number': motif_num,
            'ManualAnnotation': manual_annotation,
            'Analyzed Direction': direction.capitalize(),
            'Top Candidates': res.mapped_result if res.mapped_result is not None else "No candidates found",
            'Matrix Path': full_matrix_path if os.path.exists(full_matrix_path) else None,
            'Histogram Path': full_histo_path if os.path.exists(full_histo_path) else None,
            'Bootstrap Path': full_boot_path if os.path.exists(full_boot_path) else None,
            'Execution Time (s)': round(exec_time, 4),
            'Used Percentile': getattr(res, 'used_percentile', None),
            'Analysis Note': warnings_str.replace('<br><br>', ' | ') 
        }
        report_data.append(row_dict)
        
        # Format the HTML table row
        html_row = f"""
        <tr>
            <td><strong>{folder_name}</strong></td>
            <td><strong>{motif_num}</strong></td>
            <td>{manual_annotation}</td>
            <td>{direction.capitalize()}</td>
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
            <td>{round(exec_time, 4)}</td>
            <td>{warnings_str}</td>
        </tr>
        """
        html_lines.append(html_row)

# 5. Append clean JavaScript for Cascading Dropdowns and Table Filtering
js_script = """
<script>
const comboData = %s;

function rebuildSelect(selectId, validValues, currentValue) {
    let select = document.getElementById(selectId);
    select.innerHTML = "<option value='ALL'>All</option>";
    
    validValues.forEach(val => {
        let opt = document.createElement("option");
        opt.value = val;
        opt.textContent = val;
        if (val === currentValue) opt.selected = true;
        select.appendChild(opt);
    });
    
    if (currentValue !== 'ALL' && !validValues.includes(currentValue)) {
        select.value = 'ALL';
    }
}

function updateOptions() {
    let selectedFolder = document.getElementById("filterFolder").value;
    let selectedMotif = document.getElementById("filterMotif").value;
    let textSearch = document.getElementById("globalSearch").value.toUpperCase();

    const matchesText = (row) => {
        if (!textSearch) return true;
        const combined = (row.MemeFolder + " Motif " + row.MotifNumber).toUpperCase();
        return combined.includes(textSearch);
    };

    let folderRows = comboData.filter(row => 
        (selectedMotif === 'ALL' || row.MotifNumber === selectedMotif) &&
        matchesText(row)
    );
    let validFolders = [...new Set(folderRows.map(r => r.MemeFolder))].sort();
    
    let motifRows = comboData.filter(row => 
        (selectedFolder === 'ALL' || row.MemeFolder === selectedFolder) &&
        matchesText(row)
    );
    let validMotifs = [...new Set(motifRows.map(r => r.MotifNumber))].sort((a, b) => parseInt(a) - parseInt(b));

    rebuildSelect("filterFolder", validFolders, selectedFolder);
    rebuildSelect("filterMotif", validMotifs, document.getElementById("filterMotif").value); 
}

function filterTable() {
    const rows = document.querySelectorAll("#reportTable > tbody > tr");
    
    const folderFilter = document.getElementById("filterFolder").value.toUpperCase();
    const motifFilter = document.getElementById("filterMotif").value.toUpperCase();
    const textSearch = document.getElementById("globalSearch").value.toUpperCase();

    rows.forEach(row => {
        const folderText = row.cells[0].innerText.trim().toUpperCase();
        const motifText = row.cells[1].innerText.trim().toUpperCase();

        const matchFolder = (folderFilter === "ALL" || folderText === folderFilter);
        const matchMotif = (motifFilter === "ALL" || motifText === motifFilter);
        
        const combinedText = folderText + " Motif " + motifText;
        const matchText = (textSearch === "" || combinedText.includes(textSearch));

        if (matchFolder && matchMotif && matchText) {
            row.style.display = "";
        } else {
            row.style.display = "none";
        }
    });
}

function handleFilterChange() {
    updateOptions();
    filterTable();
}

window.onload = function() {
    updateOptions();
};
</script>
""" % (combo_json)

# %% 
html_lines.append("</tbody></table>")
html_lines.append(js_script)
html_lines.append("</body></html>")

with open(os.path.join(report_dir, f"{LABEL}_report.html"), "w") as f:
    f.write("\n".join(html_lines))

print("HTML Report generated successfully!")

# %%
# --- Generate and save the subsettable DataFrame ---

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
            new_row['evaluation'] = cand_row.get('evaluation', None)
            new_row['correctness_%'] = cand_row.get('correctness_%', None)
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
        new_row['evaluation'] = None
        new_row['correctness_%'] = None
        new_row['p_value'] = 1.0
        new_row['Inverted_Pval'] = 0.0
        expanded_rows.append(new_row)

final_report_df = pd.DataFrame(expanded_rows)

# %%
# --- Alignment with Literature ---

# 1. Create "Alignment" Column
def check_alignment(row, sig_threshold=0.05):
    pattern = str(row['ManualAnnotation']).strip().upper()
    direction = str(row['Analyzed Direction']).strip().upper()
    
    try:
        pval = float(row['p_value'])
    except (ValueError, TypeError):
        pval = 1.0
        
    # If p_value is exactly 1.0, it means no candidates were found at all
    if pval == 1.0:
        return "No candidates found"
    
    # special exception for motifs that are both DR and IR 
    if row.get('evaluation') == 'Confirmed both':
        return "Strong both" if pval < sig_threshold else "Weak both" 
        
    # Check if the analyzed direction matches the manual annotation pattern
    is_match = False
    if pattern == "IR" and direction == "REVERSE":
        is_match = True
    elif pattern == "DR" and direction == "DIRECT":
        is_match = True
    elif pattern == "BOTH":
        is_match = True
    elif pattern == "NONE":
        return "Strong disagree" if pval < sig_threshold else "Weak agree"
        
    if is_match:
        return "Strong agree" if pval < sig_threshold else "Weak agree"
    else:
        return "Strong disagree" if pval < sig_threshold else "Weak disagree"

final_report_df['Alignment'] = final_report_df.apply(check_alignment, axis=1)

# Save the master DataFrame to Excel
final_report_df.to_excel(os.path.join(report_dir, f"{LABEL}_report_data.xlsx"), index=False)
print("Master DataFrame exported to report_data.xlsx successfully!")

# 2. Extract best conclusion per observation (by min p-value)
best_idx = final_report_df.groupby(['MEME Folder', 'Motif Number'])['p_value'].idxmin()
best_conclusion_df = final_report_df.loc[best_idx].reset_index(drop=True)

best_conclusion_df.to_excel(os.path.join(report_dir, f"{LABEL}_best_conclusion_data.xlsx"), index=False)
print("Best conclusion DataFrame exported to best_conclusion_data.xlsx successfully!")

# %%
# 3. Plotting ROC and PR Curves

# Pivot the full data so every observation has both a 'Direct' and 'Reverse' continuous score
pivot_df = final_report_df.pivot_table(
    index=['MEME Folder', 'Motif Number', 'ManualAnnotation'],
    columns='Analyzed Direction',
    values='Inverted_Pval',
    aggfunc='max'
).reset_index()

# Clean up pattern strings
pivot_df['ManualAnnotation'] = pivot_df['ManualAnnotation'].astype(str).str.strip().str.upper()

# 1. Define Ground Truths (Binary 1 or 0)
y_true_dr = np.where(pivot_df['ManualAnnotation'].isin(['DR', 'BOTH']), 1, 0)
y_true_ir = np.where(pivot_df['ManualAnnotation'].isin(['IR', 'BOTH']), 1, 0)

# 2. Define Scores (Inverted P-values, higher is more confident)
y_score_dr = pivot_df['Direct'].fillna(0) if 'Direct' in pivot_df.columns else pd.Series(0.0, index=pivot_df.index)
y_score_ir = pivot_df['Reverse'].fillna(0) if 'Reverse' in pivot_df.columns else pd.Series(0.0, index=pivot_df.index)

# %%
# 3. Calculate metrics for Direct Repeats
if len(np.unique(y_true_dr)) > 1:
    fpr_dr, tpr_dr, thresh_dr = roc_curve(y_true_dr, y_score_dr)
    roc_auc_dr = auc(fpr_dr, tpr_dr)
    prec_dr, recall_dr, _ = precision_recall_curve(y_true_dr, y_score_dr)
    pr_auc_dr = average_precision_score(y_true_dr, y_score_dr)
else:
    fpr_dr, tpr_dr, thresh_dr = np.array([0, 1]), np.array([0, 1]), np.array([0, 1])
    roc_auc_dr = 0.5
    prec_dr, recall_dr = np.array([0, 1]), np.array([1, 0])
    pr_auc_dr = 0.5

# 4. Calculate metrics for Inverted Repeats
if len(np.unique(y_true_ir)) > 1:
    fpr_ir, tpr_ir, thresh_ir = roc_curve(y_true_ir, y_score_ir)
    roc_auc_ir = auc(fpr_ir, tpr_ir)
    prec_ir, recall_ir, _ = precision_recall_curve(y_true_ir, y_score_ir)
    pr_auc_ir = average_precision_score(y_true_ir, y_score_ir)
else:
    fpr_ir, tpr_ir, thresh_ir = np.array([0, 1]), np.array([0, 1]), np.array([0, 1])
    roc_auc_ir = 0.5
    prec_ir, recall_ir = np.array([0, 1]), np.array([1, 0])
    pr_auc_ir = 0.5

# Generate Plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ROC Curves
ax1.plot(fpr_dr, tpr_dr, color='blue', lw=2, label=f'Direct Repeats (AUC = {roc_auc_dr:.2f})')
ax1.plot(fpr_ir, tpr_ir, color='red', lw=2, label=f'Inverted Repeats (AUC = {roc_auc_ir:.2f})')
ax1.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title(f'{LABEL}% Threshold ROC Curves')
ax1.legend(loc='lower right')

# PR Curves
ax2.plot(recall_dr, prec_dr, color='blue', lw=2, label=f'Direct Repeats (AUC = {pr_auc_dr:.2f})')
ax2.plot(recall_ir, prec_ir, color='red', lw=2, label=f'Inverted Repeats (AUC = {pr_auc_ir:.2f})')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title(f'{THRESHOLD_MAX}% Precision-Recall Curves')
ax2.legend(loc='lower left')

plt.tight_layout()
plt.savefig(os.path.join(report_dir, f"{LABEL}_ROC_PR_Curves.png"))
plt.close()
print("ROC and PR curves saved to ROC_PR_Curves.png successfully!")

# 5. Extract Optimal Thresholds and Evaluate Each Observation
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

# Save to Excel
pivot_df.to_excel(os.path.join(report_dir, f"{LABEL}_observation_roc_evaluations.xlsx"), index=False)
print(f"Observation-level ROC evaluations exported successfully!")

# %%
# 6. More strict ROC / PR curves (Rigorous Best Conclusion Logic)

# Ensure clean text for strict evaluation
best_conclusion_df['ManualAnnotation'] = best_conclusion_df['ManualAnnotation'].astype(str).str.strip().str.upper()
best_conclusion_df['Analyzed Direction'] = best_conclusion_df['Analyzed Direction'].astype(str).str.strip().str.upper()

# Define Ground Truths (Binary 1 or 0) from the best conclusion dataframe
y_true_dr_strict = np.where(best_conclusion_df['ManualAnnotation'].isin(['DR', 'BOTH']), 1, 0)
y_true_ir_strict = np.where(best_conclusion_df['ManualAnnotation'].isin(['IR', 'BOTH']), 1, 0)

# Define Scores
y_score_dr_strict = np.where(best_conclusion_df['Analyzed Direction'] == 'DIRECT', best_conclusion_df['Inverted_Pval'], 0)
y_score_ir_strict = np.where(best_conclusion_df['Analyzed Direction'] == 'REVERSE', best_conclusion_df['Inverted_Pval'], 0)

# Calculate metrics for Rigorous Direct Repeats
if len(np.unique(y_true_dr_strict)) > 1:
    fpr_dr_s, tpr_dr_s, thresh_dr_s = roc_curve(y_true_dr_strict, y_score_dr_strict)
    roc_auc_dr_s = auc(fpr_dr_s, tpr_dr_s)
    prec_dr_s, recall_dr_s, _ = precision_recall_curve(y_true_dr_strict, y_score_dr_strict)
    pr_auc_dr_s = average_precision_score(y_true_dr_strict, y_score_dr_strict)
else:
    fpr_dr_s, tpr_dr_s, thresh_dr_s = np.array([0, 1]), np.array([0, 1]), np.array([0, 1])
    roc_auc_dr_s = 0.5
    prec_dr_s, recall_dr_s = np.array([0, 1]), np.array([1, 0])
    pr_auc_dr_s = 0.5

# Calculate metrics for Rigorous Inverted Repeats
if len(np.unique(y_true_ir_strict)) > 1:
    fpr_ir_s, tpr_ir_s, thresh_ir_s = roc_curve(y_true_ir_strict, y_score_ir_strict)
    roc_auc_ir_s = auc(fpr_ir_s, tpr_ir_s)
    prec_ir_s, recall_ir_s, _ = precision_recall_curve(y_true_ir_strict, y_score_ir_strict)
    pr_auc_ir_s = average_precision_score(y_true_ir_strict, y_score_ir_strict)
else:
    fpr_ir_s, tpr_ir_s, thresh_ir_s = np.array([0, 1]), np.array([0, 1]), np.array([0, 1])
    roc_auc_ir_s = 0.5
    prec_ir_s, recall_ir_s = np.array([0, 1]), np.array([1, 0])
    pr_auc_ir_s = 0.5

# Generate Plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Rigorous ROC Curves
ax1.plot(fpr_dr_s, tpr_dr_s, color='blue', lw=2, label=f'Direct Repeats (AUC = {roc_auc_dr_s:.2f})')
ax1.plot(fpr_ir_s, tpr_ir_s, color='red', lw=2, label=f'Inverted Repeats (AUC = {roc_auc_ir_s:.2f})')
ax1.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title(f'Rigorous ROC Curves (Best Conclusion Only)')
ax1.legend(loc='lower right')

# Rigorous PR Curves
ax2.plot(recall_dr_s, prec_dr_s, color='blue', lw=2, label=f'Direct Repeats (AUC = {pr_auc_dr_s:.2f})')
ax2.plot(recall_ir_s, prec_ir_s, color='red', lw=2, label=f'Inverted Repeats (AUC = {pr_auc_ir_s:.2f})')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title(f'Rigorous Precision-Recall Curves')
ax2.legend(loc='lower left')

plt.tight_layout()
plt.savefig(os.path.join(report_dir, f"{LABEL}_Rigorous_ROC_PR_Curves.png"))
plt.close()
print("Rigorous ROC and PR curves saved successfully!")

# 7. Evaluate True Positives/Negatives using strict p-value significance
SIG_CUTOFF = 0.05
SIG_SCORE = 1.0 - SIG_CUTOFF

def evaluate_strict_sig(score, truth, sig_threshold):
    if score >= sig_threshold:
        return 'TP' if truth == 1 else 'FP'
    else:
        return 'TN' if truth == 0 else 'FN'

best_conclusion_df[f'DR_Strict_Eval (p<={SIG_CUTOFF})'] = [evaluate_strict_sig(s, t, SIG_SCORE) for s, t in zip(y_score_dr_strict, y_true_dr_strict)]
best_conclusion_df[f'IR_Strict_Eval (p<={SIG_CUTOFF})'] = [evaluate_strict_sig(s, t, SIG_SCORE) for s, t in zip(y_score_ir_strict, y_true_ir_strict)]

# Print summaries
print(f"\n=== Strict Significance Matrix (p <= {SIG_CUTOFF}) ===")
print("Direct Repeats Evaluation:")
print(best_conclusion_df[f'DR_Strict_Eval (p<={SIG_CUTOFF})'].value_counts().to_string())
print("\nInverted Repeats Evaluation:")
print(best_conclusion_df[f'IR_Strict_Eval (p<={SIG_CUTOFF})'].value_counts().to_string())
print("===================================================\n")

# Save evaluations back to Excel
best_conclusion_df.to_excel(os.path.join(report_dir, f"{LABEL}_strict_significance_evaluations.xlsx"), index=False)
print(f"Strict significance evaluations saved to {LABEL}_strict_significance_evaluations.xlsx successfully!")


#  %%

import drippy as dp
import os
import pandas as pd
from types import SimpleNamespace
import json
import shutil

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

# %%
# 1. Setup output folders
report_dir = "MEME_OUT_REPORT"
img_dir = os.path.join(report_dir, "images")
os.makedirs(img_dir, exist_ok=True)
os.makedirs("MEME_OUTPUT", exist_ok=True)

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
        "<tr><th>MEME Folder</th><th>Motif Number</th><th>Analyzed Direction</th><th>MEME Logo</th><th>Top Candidates & P-val</th><th>Plots (Matrix, Hist, Boot)</th><th>Analysis Note</th></tr>",
        "</thead>",
        "<tbody>"
]

# Initialize empty list to store data for our final Pandas DataFrame
report_data = []

# 3. Loop through MEME motifs
for row in filepaths_dedupe.itertuples(index=False):

    folder_name = row.MemeFolder
    motif_idx = row.MotifIndex
    motif_num = row.MotifNumber
    filepath = row.Filepath
    orig_logo = row.OriginalLogo
    
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
            res = dp.detect_patterns(
                import_filepath=filepath,
                export_filepath=f"MEME_OUTPUT/{direction}_{folder_name}_motif{motif_num}",
                motif_num=motif_idx,
                direction=direction,
                metric='PIC-JSD',
                threshold_percentile=80, 
                plot_title=plot_title
            )
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
        if res.mapped_result is not None:
            all_candidates = res.mapped_result[['score', 'p_value', 'group1', 'group2']].copy()
            if pd.api.types.is_numeric_dtype(all_candidates['p_value']):
                all_candidates['p_value'] = all_candidates['p_value'].apply(lambda x: f"{x:.4e}")
            candidates_html = all_candidates.to_html(index=False).replace('\n', '')
        else:
            candidates_html = "No candidates found"
            
        parts = [getattr(res, 'threshold_note', ''), getattr(res, 'length_warning', '')]
        warnings_str = "<br><br>".join(p for p in parts if p)       

         # --- Append data for the subsettable DataFrame ---
        row_dict = {
            'MEME Folder': folder_name,
            'Motif Number': motif_num,
            'Analyzed Direction': direction.capitalize(),
            'Top Candidates': res.mapped_result if res.mapped_result is not None else "No candidates found",
            'Matrix Path': full_matrix_path if os.path.exists(full_matrix_path) else None,
            'Histogram Path': full_histo_path if os.path.exists(full_histo_path) else None,
            'Bootstrap Path': full_boot_path if os.path.exists(full_boot_path) else None,
            'Analysis Note': warnings_str.replace('<br><br>', ' | ') 
        }
        report_data.append(row_dict)
        
        # Format the HTML table row
        html_row = f"""
        <tr>
            <td><strong>{folder_name}</strong></td>
            <td><strong>{motif_num}</strong></td>
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

with open(os.path.join(report_dir, "report.html"), "w") as f:
    f.write("\n".join(html_lines))

print("HTML Report generated successfully!")

# %%
# --- Generate and save the subsettable DataFrame ---
final_report_df = pd.DataFrame(report_data)

def extract_best_candidate(candidates_data):
    if isinstance(candidates_data, pd.DataFrame) and not candidates_data.empty:
        best_idx = candidates_data['p_value'].idxmin()
        best_row = candidates_data.loc[best_idx]
        return f"Score: {best_row['score']} | P-val: {best_row['p_value']:.4e} | Groups: {best_row['group1']} / {best_row['group2']}"
    else:
        return "No candidates found"

final_report_df['Best Candidate'] = final_report_df['Top Candidates'].apply(extract_best_candidate)

final_report_df.to_csv(os.path.join(report_dir, "MEME_report_data.csv"), index=False)
print("Master DataFrame exported to report_data.csv successfully!")
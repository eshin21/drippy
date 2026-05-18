from Bio import motifs, SeqIO
from Bio.Seq import Seq
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.stats import pearsonr
import re
from types import SimpleNamespace
import warnings


# %%
########################################################################
# file processing for FAS / FASTA files 


def split_fasta_by_organism(fasta_path, output_dir, verbose = False):
    """
    Reads a multi-organism FASTA file and writes one .fas file per
    UniProt protein ID, organized into folders named by organism.

    Output structure:
        output_dir/
            Vibrio_cholerae_O1_biovar_El_Tor_str_N16961/
                TF_Fur_P0C6C8.fas
            Yersinia_pestis_CO92/
                TF_Fur_P33086.fas
    """

    def parse_header(header):
        # header arrives without the leading '>'
        # e.g. "TF_Fur_P0C6C8|genome_NC_002505.1 Vibrio cholerae O1 biovar El Tor str. N16961|start=89954|end=89974|strand=1"
        parts = header.split("|")
        tf_uniprot = parts[0].strip()                        # "TF_Fur_P0C6C8"
        
        # organism name sits between first space after 'genome_{ACCESSION}' and next '|'
        genome_field = parts[1]                              # "genome_NC_002505.1 Vibrio cholerae O1 biovar El Tor str. N16961"
        organism = genome_field.split(" ", 1)[1].strip()    # "Vibrio cholerae O1 biovar El Tor str. N16961"
        
        return tf_uniprot, organism

    def sanitize(name):
        # make filesystem-safe: replace spaces, dots, slashes with underscores
        return re.sub(r'[^\w\-]', '_', name)

    # ── accumulate records grouped by (tf_uniprot, organism) ──────────────
    groups = {}   # (tf_uniprot, organism) -> list of (header, sequence) tuples
    
    current_header = None
    current_seq_lines = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # flush previous record
                if current_header is not None:
                    tf_uniprot, organism = parse_header(current_header)
                    key = (tf_uniprot, organism)
                    if key not in groups:
                        groups[key] = []
                    groups[key].append((current_header, "".join(current_seq_lines)))

                current_header = line[1:]   # strip '>'
                current_seq_lines = []
            else:
                current_seq_lines.append(line)

        # flush final record
        if current_header is not None:
            tf_uniprot, organism = parse_header(current_header)
            key = (tf_uniprot, organism)
            if key not in groups:
                groups[key] = []
            groups[key].append((current_header, "".join(current_seq_lines)))

    # ── write output files ─────────────────────────────────────────────────
    for (tf_uniprot, organism), records in groups.items():
        folder = os.path.join(output_dir, sanitize(organism))
        os.makedirs(folder, exist_ok=True)

        out_path = os.path.join(folder, f"{tf_uniprot}.fas")
        with open(out_path, "w") as out:
            for header, seq in records:
                out.write(f">{header}\n{seq}\n")

    print(f"Done. {len(groups)} file(s) written across {len(set(o for _, o in groups))} organism folder(s).")
    if verbose:
        return groups  # return if caller wants to inspect without touching disk
    else:
        return

########################################################################

# %% 
# construct motif 


def load_motif(filename, motif_num=0):
    """
    Returns a genuine Bio.motifs Motif object regardless of input format.
    
    For .xml / MEME: parses directly, returns motifsM[motif_num]
    For .fas / .fasta: reads sequences, constructs a real Motif object
    
    Downstream functions (make_ppm, map_back, etc.) receive the same
    object type either way — no shims, no fake interfaces.
    """

    ext = filename.lower()

    if ext.endswith(".xml"):
        with open(filename) as handle:
            motifsM = motifs.parse(handle, "meme")
        return motifsM[motif_num]

    elif ext.endswith((".fasta", ".fas")):
        sequences = []
        with open(filename) as handle:
            for record in SeqIO.parse(handle, "fasta"):
                sequences.append(record.seq)

        lengths = [len(s) for s in sequences]
        if len(set(lengths)) > 1:
            min_len = min(lengths)
            sequences = [s[:min_len] for s in sequences]
            warnings.warn(
                f"Inconsistent sequence lengths {set(lengths)} in '{filename}'; "
                f"trimmed all to {min_len}bp.",
                UserWarning
            )

        motif = motifs.create(sequences, alphabet="ACGT")
        return motif
    

    else:
        raise ValueError(f"Unsupported file type: {filename}")


# %%
########################################################################
# Compute PPM and complement PPM
## input = Motif object
## existing Biomotif motif.pwm is a custom Bio.motifs.matrix.PositionWeightMatrix type, they perform their own smoothing / adjustment -- want to stay as pure as possible to the raw data 

########################################################################
def make_ppm(motif, pseudocount=1e-10):

    aligned_seq_matrix = []
    for i in motif.alignment.sequences:
        aligned_seq_matrix.append(list(str(i)))
        
    df_seq = pd.DataFrame(aligned_seq_matrix)


    # make running storage for each base for each column index
    # our new data frame will have 12 columns and 4 rows 
    A_tally = []
    C_tally = []
    G_tally = []
    T_tally = []

    for column_index in range(df_seq.shape[1]): 
        current_column = list(df_seq[column_index])

        A_count = current_column.count("A") # how many A's when counting down columnwise
        C_count = current_column.count("C")
        G_count = current_column.count("G")
        T_count = current_column.count("T")

        A_tally.append(A_count)
        C_tally.append(C_count)
        G_tally.append(G_count)
        T_tally.append(T_count)

    # rename columns    with dictionart
    psfm_data = {
        'A': A_tally,
        'C': C_tally,
        'G': G_tally,
        'T': T_tally
    }

    psfm = pd.DataFrame(psfm_data)
    # flip transpose so that each row is A, C, G T and columns are positions 
    psfm = psfm.transpose()
    # probability matrix (counts in each column divided by number of sequences)
    ppm = pd.DataFrame(psfm / len(df_seq))

    ppm_np = ppm.to_numpy() # use np array for computation, but use DF later for visualization since DF rows and columns can be labeled
    
    # Introduce pseudocount globally and re-normalize columns to sum to 1
    ppm_np = ppm_np + pseudocount
    ppm_np = ppm_np / ppm_np.sum(axis=0)
    
    return ppm_np

def complement_ppm(ppm_np):
    return(
        ppm_np[[3, 2, 1, 0], :]
    )


# %%



########################################################################
# Pairwise Comparison 
## input = metric type
########################################################################
def compute_metrics(ppm_np, metric = 'PIC-JSD', direction = 'direct'):

    if(metric == 'PIC-JSD'):
        res = positional_information_content(ppm_np, direction=direction) -  jensen_shannon(ppm_np, direction=direction)
    elif(metric == 'PIC'):
        res = positional_information_content(ppm_np, direction=direction)
    elif(metric == 'JSD'):
        res = jensen_shannon(ppm_np, direction=direction)

    elif(metric == 'Pearson'):
        res = pearson(ppm_np, direction=direction)
    else:
        raise ValueError(f"Unknown metric provided: {metric}")

    # TODO: PIC*Pearson

    return res


########################################################################
# Metric: IC
########################################################################

def positional_information_content(ppm_np, direction='direct', bg_probs_dict = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}):

    # get complemented PPM
    if (direction == 'reverse'):
        comp_ppm = complement_ppm(ppm_np)
    
    # convert dictionary to array 
    bg_probs = np.array([bg_probs_dict[b] for b in ['A', 'C', 'G', 'T']])
    H_before = 0 
    
    # compute prior information 
    for i in range(len(bg_probs)):
        p = bg_probs[i]
        term = p * np.log2(p)
        H_before = H_before + term
    H_before = -H_before
    

    # Calculations 
    
    ### storage
    num_positions = ppm_np.shape[1]
    after_matrix = np.zeros((num_positions, num_positions))

    for i in range(num_positions):
        x = ppm_np[:, i]  

        for j in range(num_positions):
            if i == j and direction == 'direct': 
                after_matrix[i, j] = 0 
                continue # we dont need to do identity 
            if i != j and direction == 'direct':
                y = ppm_np[:, j]
            else:
                if(direction == 'reverse'):
                    y = comp_ppm[:, j]  ## key difference here -- we have to use the complement matrix. We compare reverse from an outwards-in fashion, unlike the direct repeat where we do pairwise from left to right
            
            xy = (x + y) / 2 
            H_after = -sum(xy * np.log2(xy))
            after_matrix[i, j] = H_after

    info_content_res = H_before - after_matrix
    
    return info_content_res


########################################################################
# Metric: JSD
########################################################################
# 
# %%
def jensen_shannon(ppm_np, direction='direct'):
    
    # number of columns(indices)
    num_positions = ppm_np.shape[1]

    # storage for results as a 2D numpy array
    jsd_results = np.zeros((num_positions, num_positions))

    if direction == 'reverse':
        comp_ppm = complement_ppm(ppm_np)

    for i in range(num_positions):
        x = ppm_np[:, i]

        for j in range(num_positions):

            if direction == 'direct':
                y = ppm_np[:, j]
            elif direction == 'reverse':
                y = comp_ppm[:, j]
                
            midpoint = (x + y) / 2 
            D_XM = sum(x * np.log2(x / midpoint))
            D_YM = sum(y * np.log2(y / midpoint))
            JSD = 0.5 * (D_XM + D_YM)
            jsd_results[i, j] = JSD
            
    return jsd_results
# %%


########################################################################
# Metric: Pearson
########################################################################

def pearson(ppm_np, direction='direct'):

    # get complemented PPM
    if (direction == 'reverse'):
        comp_ppm = complement_ppm(ppm_np)

    # number of columns(indices)
    num_positions = ppm_np.shape[1]

    # storage for results as a 2D numpy array
    pearson_results = np.zeros((num_positions, num_positions))


    for i in range(num_positions):
        x = ppm_np[:, i]

        for j in range(num_positions):
            if(i == j): 
                pearson_results[i, j] = 0 
                continue # we dont need to do identity
            elif(direction=='direct'):
                    y = ppm_np[:, j]
                    pearson_results[i, j] = pearsonr(x, y)[0]
            elif(direction=='reverse'):
                    y = comp_ppm[:, j]
                    pearson_results[i, j] = pearsonr(x, y)[0]

    return pearson_results
            
########################################################################
# Diagonal scoring based on threshold
## input = metric matrix, threshold 

def score_diagonals(matrix, threshold, direction='direct'):
    n = matrix.shape[0]
    all_candidates = []

    for start_i in range(n):
        for start_j in range(start_i): # check below main diagonal. axis of symmetry same in both direct and RC 

            i = start_i
            j = start_j
            coords = []
            score_sum = 0 # reset it for each new starting position

            while 0 <= i < n and 0 <= j < n:
                if matrix[i, j] >= threshold:
                    coords.append((i, j))
                    score_sum += matrix[i, j]
                    # print(coords)
                    i += 1
                    j += 1 if direction == 'direct' else -1   
                else:
                    break
            
            # Save all diagonals greater or eq than length 2
            if len(coords) >= 2: 
                all_candidates.append({
                    "coords": coords,
                    "length": len(coords),
                    "score": score_sum
                })

    # Filter out sub-diagonals
    # Sort by length descending so we evaluate the longest diagonals first
    all_candidates.sort(key=lambda x: x["length"], reverse=True)
    
    filtered_candidates = []
    for candidate in all_candidates:
        c_set = set(candidate["coords"])
        # Check if this candidate is a subset of any already-kept longer diagonal
        is_subset = any(c_set.issubset(set(kept["coords"])) for kept in filtered_candidates)
        if not is_subset:
            filtered_candidates.append(candidate)

    # return all_candidates
    return filtered_candidates
# %%
##################################################################
## shuffling -- null distribution for p-values 
##################################################################

# for doing statistical testing of our found diagonals, we can shuffle two ways
# 1. the rows and columns are both shuffled independently.
# 2. we shuffle only the indices, but apply the same shuffling to the rows and columns. This option is most logical as the goal is to destroy index-wise relationships.

def shuffle_metrics(metrics_matrix, myseed = 42):
    
    # set seed by creating a random number generator (rng)
    rng = np.random.default_rng(seed=myseed)

    # number of positions 
    n = metrics_matrix.shape[0]
    
    # 3. Create a random permutation of the INDICES (option 2)
    indices = rng.permutation(n)

    # 4. Apply the same shuffled indices to both rows and columns
    shuffled_matrix = metrics_matrix[np.ix_(indices, indices)]

    return shuffled_matrix

# %%

##################################################################
## bootstrap construction
# 2. we shuffle only the indices, but apply the same shuffling to the rows and columns. This option is most logical as the goal is to destroy index-wise relationships.
##################################################################

def bootstrap_scores(metrics_matrix, myseed=42, iterations=1000, threshold=1.0, direction='direct'):
    
    #  storage for  null distribution
    bootstrap_scores = []

    for i in range(iterations):
        # change seed each iteration to ensure unique shuffles
        shuffled = shuffle_metrics(metrics_matrix, myseed=myseed + i)
        
        # re-score shuffled matrix  -- carry consistent threshold  
        dia = score_diagonals(shuffled, threshold=threshold, direction=direction)
        
        # Save only the top score (highest score) from this iteration, default to 0 if none found
        top_score = max((candidate["score"] for candidate in dia), default=0)            
        bootstrap_scores.append(top_score)

    return bootstrap_scores

# %%

########################################################################
# Visualizations
########################################################################
# %%


def visualize_matrix(input_matrix, colorscheme = 'viridis', lowerbound = -1, upperbound = 2, title = None, flip_rows=False):

    display_matrix = np.array(input_matrix.copy(), dtype=float)
    num_positions = display_matrix.shape[0]
    
    # for reverse complement, we manually do a flip of one of the axes to force the diagonal runs to be in the same direction as the direct repeats. But we also have to fix the axis labels in the visualization to reflect the flip 
    if flip_rows:
        row_labels = list(range(num_positions - 1, -1, -1))
        display_matrix = display_matrix[::-1, :]
    else:
        row_labels = list(range(num_positions))


    fig = plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        display_matrix, 
        annot=True,       # Turn on if you want to see the numbers
        annot_kws={"size":7},
        fmt='.2f',
        cmap=colorscheme,
        vmin=lowerbound, vmax=upperbound,    
        square=True,
        yticklabels=row_labels
    )
    ax.set_facecolor('white')    #NA cells white 
    plt.title(title)
    plt.xlabel("Motif Position")
    plt.ylabel("Motif Position")
    plt.close()
    return fig

# %%

def histogram_scores(input_np, title =  "Distribution of Scores", top_score=None, top_score_label="Top Score"):
    
    # 1. Flatten the matrix to a 1D array so every cell is treated as a single data point
    # We use .to_numpy() to ensure it's a math-ready array, then .flatten()
    all_values = input_np.flatten()

    # 2.  Create the distribution plot
    fig = plt.figure(figsize=(8, 5))
    sns.histplot(all_values, kde=True, bins=30, color='skyblue')

    if top_score is not None:
        plt.axvline(x=top_score, color='red', linestyle='dashed', linewidth=2, label=f'{top_score_label}: {top_score:.2f}')
        plt.legend()

    # 3. Add labels and title
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.close(fig)
    return fig 


########################################################################
# Utilities 
######################################################################### %%

def map_back(motif, candidates):
 
 # get the candidates and "map them back" to the indices of the consensus 
# decode the candidates into a pair of nucleotide patterns
# return array candidates with two new columns 
#  group1: corresponding nucleotides based on indices in consensus string, where indices are denoted by the row number in each pair of coordinates 
# group2: same but for columns 

    #built in biomotif property 
    consensus = str(motif.consensus)

    if any(base not in "ACGT" for base in consensus):
        print(f"Warning: Consensus sequence '{consensus}' contains ambiguous IUPAC base codes.")
    
    # each candidate is a diagonal >= 2bp 

    for candidate in candidates:
        coords = candidate['coords']
        
        group1_str = ""
        group2_str = ""
        
        # get nucleotide in consensus corresponding to row (r) and column (c) indices from candidate list 
        # goal is to output columns group1 ...group2 like ATCG...ATCG

        for r, c in coords:
            group1_str += consensus[r]
            group2_str += consensus[c]
            
        candidate['group1'] = group1_str
        candidate['group2'] = group2_str
        
    # Convert to DF for display
    return pd.DataFrame(candidates)


# %% 
########################################################################
# Compute better thresholding using percentiles, not a fixed constant  
########################################################################


def thresholder(metrics, percentile=75):
    flat_metrics = metrics.flatten()
    return np.percentile(flat_metrics, percentile)



# %%


########################################################################
# Orchestrator
########################################################################


def detect_patterns(import_filepath, export_filepath, direction = 'direct', metric = 'PIC-JSD', threshold_percentile = 80, min_threshold_percentile = 25, fallback_step = 5, minbackground = None, plot_title = None, bootstrap_iterations = 5000):
    
    # 1.  File IO, create PPM and Motif object

    # capture sencarios where the sequences are not of the same length

    length_warning = ""  # will remain empty if sequences are all the same length
   

   #enter and exit warning capture for mismatch length .fas files   
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        motif = load_motif(import_filepath)

    motif = load_motif(import_filepath)               #  run — any warn() goes into caught


    if warning_list:
        length_warning = str(warning_list[0].message)


    ppm =  make_ppm(motif)
    
    # 2. Compute metrics matrix 
    metrics = compute_metrics(ppm, metric=metric, direction=direction)

    # 3. Compute IC-based threshold
    pic = compute_metrics(ppm, metric='PIC', direction=direction)

    mythreshold = thresholder(pic, percentile=threshold_percentile)

    # preserve original threshold
    original_threshold = mythreshold.copy()

    # empty override title for automatically adjusted threshold

    override_plot_title = ""

    threshold_note = ""

    # 4. Diagonal candidates and map back to get readable base pair segments   
    candidates = [] # empty intial
    pct_used_threshold = threshold_percentile
    
    candidates = score_diagonals(metrics, threshold = mythreshold, direction=direction)

    #4a: sometimes, the specified percentile threshold is too high, and there are no diagonals (>=2 cell runs) at all that meet the threshold to be valid candidates from score_diagonals. So, we add logic to iteratively decrement the user-defined threshold automatically.

    while not candidates:

        # Decrement safely without overstepping the minimum limit and retry 
        
        pct_used_threshold = max(pct_used_threshold - fallback_step, min_threshold_percentile)


        print(f"[DETECT_PATTERNS] - {plot_title}: No candidates at previous threshold, trying {pct_used_threshold}%...")
        mythreshold = thresholder(pic, percentile=pct_used_threshold)
        candidates = score_diagonals(metrics, threshold=mythreshold, direction=direction)
            

          # total failure case - can't go any lower 
            ## construct an exception object with note 
            
        if pct_used_threshold <= min_threshold_percentile:
          
            none_msg = (f"[DETECT_PATTERNS] - {plot_title}: Exhausted all fallbacks down to {min_threshold_percentile}th percentile. No diagonal candidates >=2 positions found at user-specified {threshold_percentile}%")

            fig_hist = histogram_scores(
                metrics,
                title=f"[*** Exhausted fallbacks down to {min_threshold_percentile}%] \n Distribution of {metric} Metrics, Direction {direction} \n {plot_title}",
                top_score=original_threshold,
                top_score_label=f"{pct_used_threshold}th Percentile")

            fig_matrix = visualize_matrix(
                metrics,
                title=f"Matrix of {metric} Scores, Direction {direction} \n {plot_title}")

            print(none_msg)

            return SimpleNamespace(
                motif = motif,
                pval = None,
                metrics = metrics,
                threshold = original_threshold,
                candidates = None,
                mapped_result = None,
                plots={'histogram': fig_hist, 'matrix': fig_matrix, 'bootstrap': None},
                threshold_note = none_msg,
                length_warning=length_warning
            )



    if pct_used_threshold != threshold_percentile:

        override_plot_title = f'[*** Overrode User Percentile to {pct_used_threshold}%] \n'

        threshold_note = f"[{plot_title}]: No diagonal candidates were possible at the selected {threshold_percentile}% threshold for {import_filepath}, automatically reduced to {pct_used_threshold}%, at value {mythreshold}."
    
        print(f"[DETECT_PATTERNS] - {plot_title} Fell back to {pct_used_threshold}th percentile (threshold: {mythreshold:.4f})")


    mapped_result = map_back(motif, candidates)

    # 5. bootstrapping
    boot = bootstrap_scores(metrics, myseed=42, iterations=bootstrap_iterations, threshold=mythreshold, direction=direction)

    
    # 6. Compute p-values  for all candidates 

    boot_array = np.array(boot)
    p_values = []
    
    for candidate in candidates:
        c_score = candidate["score"]
        # count proportion of values that are geq than observed candidate score
        c_p_value = np.sum(boot_array >= c_score) / len(boot_array)
        p_values.append(c_p_value)
        
    mapped_result["p_value"] = p_values
    mapped_result.to_excel(f"{export_filepath}.xlsx")

    # get top scores
    top_score = max(candidate["score"] for candidate in candidates)
    p_value = np.sum(boot_array >= top_score) / len(boot_array)
    print(f"Computed p-value for top score: {p_value:.5e}")


    # Visualizations of scores, matrix, bootstrapping


    fig_hist = histogram_scores(
        metrics,
        title=f"{override_plot_title} \n Distribution of {metric} Metrics, Direction {direction} \n {plot_title}",
            top_score=mythreshold,
            top_score_label=f"{pct_used_threshold}th Percentile")


    fig_matrix = visualize_matrix(
        metrics,
        title=f"Matrix of {metric} Scores, Direction {direction} \n {plot_title}")

    fig_boot = histogram_scores(
        np.array(boot),
        title=f"Distribution of Bootstrapped Top Scores, Direction {direction}, \n{plot_title} (p={round(p_value, 5)})",
        top_score=top_score)

    return SimpleNamespace(
        motif=motif,
        pval = p_value,
        metrics=metrics,
        threshold=mythreshold,
        candidates=candidates,
        mapped_result=mapped_result,
        plots={'histogram': fig_hist, 'matrix': fig_matrix,
               'bootstrap': fig_boot},
        threshold_note = threshold_note,
        length_warning=length_warning

    )

# %%

def filename_to_title(filepath):
    

    # filepath = 'CollecTF_FASTA/LexA/Paracoccus_denitrificans_PD1222/TF_LexA_A1B3Z0.fas'
    species_fas_folder = filepath
    
    # split by filepath by / char, keep objects 2:3 for species name, UniProtID  

    # e.g.'CollecTF_FASTA/LexA/Paracoccus_denitrificans_PD1222/TF_LexA_A1B3Z0.fas'
    ## becomes   Paracoccus_denitrificans_PD1222
    family = species_fas_folder.split("/")[1:2]
    species_uid = species_fas_folder.split("/")[2:3]

    return(family, species_uid)

    

# %% 



if __name__ == "__main__":



    ###################################################### 
    ### FILE I/O Accessing CollecTF .FAS files
    ######################################################

# Q8PN77  -- single sample example -- test case 
# C1F978 -- 2-sample case 

# biopython weblogo call + human review call + results and p-value 
# negative examples -- we can either generate ourselves 
# ignasi -- ingest FASTA files without forced IR pattern 

    # our usage only

    # split_fasta_by_organism("IMPORTS/FNR_CRP_collectf-export-fasta.fas", output_dir="CollecTF_FASTA/FNR_CRP")


    # %%
                
        
    family = 'LexA'
    species_fas_folder = 'Rhodobacter_sphaeroides_2_4_1/TF_LexA_C1F978.fas'

    
    # split by filepath by _ char, keep objects 0,1 for species name, rejoin  

    # e.g.  Xanthomonas_axonopodis_pv__citri_str__306/TF_LexA_Q8PN77.fas
    ## becomes   Xanthomonas_axonopodis
    species = '_'.join(species_fas_folder.split("_")[0:2])

    # split filepath by / char, get the isolated .fas name 
    ## e.g.  Xanthomonas_axonopodis_pv__citri_str__306/TF_LexA_Q8PN77.fas
    ### becomes ['TF', 'LexA', 'Q8PN77.fas']

    fasname = species_fas_folder.split("/")[1].split("_")

    ## extract UniProt_ID
    ### e.g  ['TF', 'LexA', 'Q8PN77.fas'] becomes Q8PN77
    keyname = re.sub('[.fas]', '', fasname[2])

    direction = 'reverse'


    # %% 
    res = detect_patterns(
        import_filepath = f"CollecTF_FASTA/{family}/{species_fas_folder}", 
        export_filepath = f"OUTPUT/{family}/{direction}_{species}_{keyname}",
        direction = direction,
        metric = 'PIC-JSD',
        threshold_percentile = 80, 
        plot_title = f"{species}_{keyname}", 
        bootstrap_iterations = 5000
        )




    res.plots['histogram']
    res.plots['matrix']
    res.plots['bootstrap']
    pd.DataFrame(res.candidates)

    res.motif
    res.motif.weblogo(f"{keyname}.png", format = 'png')





    # %%

    import_filepath = f"CollecTF_FASTA/{family}/{species_fas_folder}"

    export_filepath = f"OUTPUT/{family}/{direction}_{species}_{keyname}"
    direction = 'reverse'

    metric = 'PIC-JSD'
    bootstrap_iterations = 5000

    plot_title = f"{species}_{keyname}"
    

    # 1.  File IO, create PPM and Motif object
    motif = load_motif(import_filepath)
    ppm =  make_ppm(motif)
    
    # motif.weblogo("test.png", format='png')

    # 2. Compute metrics matrix 
    metrics = compute_metrics(ppm, metric=metric, direction=direction)
    visualize_matrix(metrics, title=import_filepath)

    # 3. Compute IC-based threshold
    pic = compute_metrics(ppm, metric='PIC', direction=direction)
    
    mythreshold = thresholder(pic, percentile=80)  
    
    histogram_scores(metrics)


    # 4. Diagonal candidates and map back to get readable base pair segments   
    
    candidates = score_diagonals(metrics, threshold = mythreshold, direction=direction)
    mapped_result = map_back(motif, candidates)

    # 5. bootstrapping
    boot = bootstrap_scores(metrics, myseed=42, iterations=bootstrap_iterations, threshold=mythreshold, direction=direction)

    
    # 6. Compute p-values  for all candidates 

    boot_array = np.array(boot)
    p_values = []
    
    for candidate in candidates:
        c_score = candidate["score"]
        # count proportion of values that are geq than observed candidate score
        c_p_value = np.sum(boot_array >= c_score) / len(boot_array)
        p_values.append(c_p_value)
        
    mapped_result["p_value"] = p_values
    mapped_result.to_excel(f"{export_filepath}.xlsx")

    print(candidates)
    # get top scores
    top_score = max(candidate["score"] for candidate in candidates)
    p_value = np.sum(boot_array >= top_score) / len(boot_array)
    print(f"Computed p-value for top score: {p_value:.5e}")


    # Visualizations of scores, matrix, bootstrapping
    fig_hist = histogram_scores(
        metrics,
        title=f"Distribution of {metric} Metrics, Direction {direction} \n {plot_title}",
        top_score=mythreshold,
        top_score_label=f"80th Percentile")

    fig_matrix = visualize_matrix(
        metrics,
        title=f"Matrix of {metric} Scores, Direction {direction} \n {plot_title}")

    fig_boot = histogram_scores(
        np.array(boot),
        title=f"Distribution of Bootstrapped Top Scores, Direction {direction}, \n{plot_title} (p={round(p_value, 5)})",
        top_score=top_score)

    # direction = 'reverse'
    # title = 'Staph_LexA_Q9L4P1'

    # ic_jsd = compute_metrics(lexA_ppm, metric='PIC-JSD', direction="reverse")

    # pic = compute_metrics(lexA_ppm, metric='PIC', direction="reverse")

    # mythreshold = thresholder(pic, percentile=80); print(mythreshold)

    # histogram_scores(ic_jsd, title=f"Distribution of PIC-JSD Metrics, Direction {direction} \n  LexA Staph Motif Q9L4P1",top_score=mythreshold, top_score_label="80th Percentile")

    # candidates = score_diagonals(ic_jsd, threshold = mythreshold, direction=direction)

    # visualize_matrix(ic_jsd, colorscheme='viridis', lowerbound=-1, upperbound=2, title=f"{title}, {direction}", flip_rows=False)

    # mapped_result = map_back(motif_lexA, candidates)

    # mapped_result.to_excel(f"OUTPUT/LexA/{title}.xlsx")


# %%

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
        
        if len(sequences) < 2:
            raise ValueError(f"'{filename}' contains only {len(sequences)} sequence(s); skipping.")

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
def make_ppm(filename, pseudocount=1e-10):

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
def compute_metrics(ppm_np, metric = 'PIC-JSD', direction = 'main'):

    if(metric == 'PIC-JSD'):
        res = positional_information_content(ppm_np, direction=direction) -  jensen_shannon(ppm_np, direction=direction)

    if(metric == 'PIC'):
        res = positional_information_content(ppm_np, direction=direction)

    if(metric == 'JSD'):
        res = jensen_shannon(ppm_np, direction=direction)

    # TODO: PIC*Pearson

    return res


########################################################################
# Metric: IC
########################################################################

def positional_information_content(ppm_np, direction='main', bg_probs_dict = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}):

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
            if i == j and direction == 'main': 
                after_matrix[i, j] = 0 
                continue # we dont need to do identity 
            if i != j and direction == 'main':
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
def jensen_shannon(ppm_np, direction='main'):
    
    # number of columns(indices)
    num_positions = ppm_np.shape[1]

    # storage for results as a 2D numpy array
    jsd_results = np.zeros((num_positions, num_positions))

    if direction == 'reverse':
        comp_ppm = complement_ppm(ppm_np)

    for i in range(num_positions):
        x = ppm_np[:, i]

        for j in range(num_positions):

            if direction == 'main':
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

def pearson(ppm_np, direction='main'):

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
            elif(direction=='main'):
                    y = ppm_np[:, j]
                    pearson_results[i, j] = pearsonr(x, y)[0]
            elif(direction=='reverse'):
                    y = comp_ppm[:, j]
                    pearson_results[i, j] = pearsonr(x, y)[0]

    return pearson_results
            
########################################################################
# Diagonal scoring based on threshold
## input = metric matrix, threshold 

def score_diagonals(matrix, threshold, direction='main'):
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
                    j += 1 if direction == 'main' else -1   
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

def bootstrap_scores(metrics_matrix, myseed=42, iterations=1000, threshold=1.0, direction='main'):
    
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


def visualize_matrix(input_matrix, colorscheme, lowerbound, upperbound, title, flip_rows=False):

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

# Compute better thresholding using percentiles, not a fixed constant  
def thresholder(metrics, percentile=75):
    flat_metrics = metrics.flatten()
    return np.percentile(flat_metrics, percentile)

# %%

def detect_patterns(import_filepath, export_filepath, direction = 'main', metric = 'PIC-JSD', threshold_percentile = 80, background = None, plot_title = None, bootstrap_iterations = 5000):
    
    # 1.  File IO, create PPM and Motif object
    motif = load_motif(import_filepath)
    ppm =  make_ppm(motif)
    
    # 2. Compute metrics matrix 
    metrics = compute_metrics(ppm, metric={metric}, direction={direction})

    # 3. Compute IC-based threshold
    pic = compute_metrics(ppm, metric='PIC', direction={direction})
    mythreshold = thresholder(pic, percentile={threshold_percentile})  
    
    # 4. Diagonal candidates
    
    candidates = score_diagonals(ic_jsd, threshold = mythreshold, direction=direction)
    mapped_result = map_back(motif, candidates)
    mapped_result.to_excel(f"{export_filepath}.xlsx")


    # 5. bootstrapping
    boot = bootstrap_scores(metrics, myseed=42, iterations=bootstrap_iterations, threshold=mythreshold, direction=direction)


    # Visualizations of scores, matrix, bootstrapping
    fig_hist = histogram_scores(
        metrics,
        title=f"Distribution of {metric} Metrics, Direction {direction} \n {title}",
        top_score=mythreshold,
        top_score_label=f"{threshold_percentile}th Percentile")

    fig_matrix = visualize_matrix(
        metrics,
        title=f"Matrix of {metric} Scores, Direction {direction} \n {title}",
        direction={direction})

    fig_boot = histogram_scores(
        np.array(boot),
        title=f"Distribution of Bootstrapped Top Scores, \n{title} (p={round(p_value, 5)})",
        top_score=top_score)


    return SimpleNamespace(
        metrics=metrics,
        threshold=mythreshold,
        candidates=candidates,
        mapped_result=mapped_result,
        plots={'histogram': fig_hist, 'matrix': fig_matrix,
               'bootstrap_histogram': fig_boot}
    )


if __name__ == "__main__":



    ###################################################### 
    ### FILE I/O Accessing CollecTF .FAS files
    ######################################################


    # %%
    # our usage only

    split_fasta_by_organism("IMPORTS/LexA_collectf-export-fasta.fas", output_dir="CollecTF_FASTA/LexA")

    # FASTA (already split into single-organism files by split_fasta_by_organism)
    motif_lexA = load_motif("CollecTF_FASTA/LexA/Staphylococcus_aureus_subsp__aureus_COL/TF_LexA_Q9L4P1.fas")
    ppm = make_ppm(motif_lexA)

    # %%

    direction = 'reverse'
    title = 'Staphylococcus_aureus_subsp__aureus_COL TF_LexA_Q9L4P1'

    ic_jsd = compute_metrics(ppm, metric='PIC-JSD', direction="reverse")

    pic = compute_metrics(ppm, metric='PIC', direction="reverse")

    mythreshold = thresholder(pic, percentile=80); print(mythreshold)


    histogram_scores(ic_jsd, title=f"Distribution of PIC-JSD Metrics, Direction {direction} \n  LexA Staph Motif Q9L4P1",top_score=mythreshold, top_score_label="80th Percentile")

    candidates = score_diagonals(ic_jsd, threshold = mythreshold, direction=direction)

    visualize_matrix(ic_jsd, colorscheme='viridis', lowerbound=-1, upperbound=2, title=f"{title}", flip_rows=False)

    mapped_result = map_back(motif_lexA, candidates)

    mapped_result.to_excel(f"OUTPUT/LexA/{title}.xlsx")



    # %%
    ###################################################### 
    ### WORKING WITH MOTIF XML objects
    ######################################################

    ex = 4 
    motif_num = 2
    direction = 'reverse'
    meme_file = f"IMPORTS/meme_out_{ex}/meme.xml"


    # XML
    motif = load_motif(meme_file, motif_num=0)
    ppm = make_ppm(motif)

    #%% 
    
    ic_jsd = compute_metrics(ppm, metric='PIC-JSD', direction=direction)

    pic = compute_metrics(ppm, metric='PIC', direction=direction)

    mythreshold = thresholder(pic, percentile=80); print(mythreshold)


    histogram_scores(ic_jsd, title=f"Distribution of PIC-JSD Metrics, Direction {direction} \n  Ex{ex} Motif {motif_num+1}",top_score=mythreshold, top_score_label="80th Percentile")

    candidates = score_diagonals(ic_jsd, threshold = mythreshold, direction=direction)

    pd.DataFrame(candidates)
    #%%
    visualize_matrix(ic_jsd, colorscheme='viridis', lowerbound=-1, upperbound=2, title=f"Ex{ex} Motif {motif_num+1}: Information-JSD, Direction {direction}", flip_rows=False)

    mapped_result = map_back(motif, candidates)

    mapped_result.to_excel(f"OUTPUT/{direction}_Ex_{ex}_Motif_{motif_num+1}.xlsx")


  
    # %%
    # BOOTSTRAPPING

    n_iter = 5000

    boot = bootstrap_scores(ic_jsd, myseed=42, iterations=n_iter, threshold=mythreshold, direction=direction)
    
    # %%

    top_score = max(candidate["score"] for candidate in candidates)
    # count proportion of values that are geq than observed top score 

    print(len(boot))

    p_value = np.sum(np.array(boot) >= top_score) / len(boot)
    print(f"Computed p-value: {p_value}")

    histogram_scores(np.array(boot), title=f"Distribution of Bootstrapped Top Scores, Direction {direction} \n Ex{ex} Motif {motif_num+1} (p={round(p_value, 5)})", top_score=top_score)


# %%    

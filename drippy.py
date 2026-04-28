from Bio import motifs
from Bio import SeqIO 
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns




########################################################################
# Compute PPM and complement PPM
## input = BioMotif
## existing Biomotif motif.pwm is a custom Bio.motifs.matrix.PositionWeightMatrix type which is proprietary
########################################################################

# %%
def make_ppm(motif):
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
                    y = comp_ppm[:, j]  ## key difference here -- we have to use the complement matrix. We compare reverse from an outwwards-in fashion, unlike the direct repeat where we do right to left pair
            
            xy = (x + y) / 2 
            H_after = -sum(xy * np.log2(xy + 1e-10)) #add a pseudocount because some values a0, log0 NaN
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
        x = ppm_np[:, i] + 1e-10

        for j in range(num_positions):

            if direction == 'main':
                y = ppm_np[:, j] + 1e-10
            elif direction == 'reverse':
                y = comp_ppm[:, j] + 1e-10
                
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
        x = ppm_np.iloc[:, i]

    for j in range(num_positions):
        if(i == j): 
            pearson_results.iloc[i, j] = 0 
            continue # we dont need to do identity
        elif(direction=='main'):
                y = ppm_np.iloc[:, j]
                pearson_results.iloc[i, j] = pearsonr(x, y)[0]
        elif(direction=='reverese'):
                y = comp_ppm[:, j]
                pearson_results.iloc[i, j] = pearsonr(x, y)[0]

    return pearson_results
            
########################################################################
# Diagonal scoring based on threshold
## input = metric matrix, threshold 
########################################################################

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
## every day im shuffling -- null distribution for p-values 
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
##################################################################

def scoring_bootstrap(metrics_matrix, myseed=42, iterations=1000, threshold=1.0, direction='main'):
    
    # Initialize storage for our null distribution
    bootstrap_scores = []

    for i in range(iterations):
        # Use a dynamic seed based on the iteration to ensure unique shuffles
        shuffled = shuffle_metrics(metrics_matrix, myseed=myseed + i)
        
        # Score the shuffled matrix (not the global ic_jsd!)
        dia = score_diagonals(shuffled, threshold=threshold, direction=direction)
        
        if dia:
            # Save only the top score (highest score) from this iteration
            top_score = max(candidate["score"] for candidate in dia)
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


    plt.figure(figsize=(10, 8))
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
    plt.show()

# %%

def histogram_scores(input_np, title =  "Distribution of Scores", top_score=None):
    
    # 1. Flatten the matrix to a 1D array so every cell is treated as a single data point
    # We use .to_numpy() to ensure it's a math-ready array, then .flatten()
    all_values = input_np.flatten()

    # 2.  Create the distribution plot
    plt.figure(figsize=(8, 5))
    sns.histplot(all_values, kde=True, bins=30, color='skyblue')

    if top_score is not None:
        plt.axvline(x=top_score, color='red', linestyle='dashed', linewidth=2, label=f'Top Score: {top_score:.2f}')
        plt.legend()

    # 3. Add labels and title
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.show()

# %%



# ONLY things inside this block run when you run drippy.py directly.
# Everything inside here is SKIPPED when you "import drippy" elsewhere. # %%

if __name__ == "__main__":


    meme_file = "IMPORTS/meme_out_3/meme.xml"

    ###################################################### 
    ### FILE I/O Accessing motif objects
    ######################################################
    with open(meme_file) as handle:
        motifsM = motifs.parse(handle, "meme")

    i = 3 ## TTCC...GGAA
    motif = (motifsM)[i]
    ppm = make_ppm(motif)
    
    ic_jsd = compute_metrics(ppm, metric='PIC-JSD', direction='reverse')

    dia = score_diagonals(ic_jsd, threshold = 1.2, direction='reverse')
    pd.DataFrame(dia)


    visualize_matrix(ic_jsd, colorscheme='viridis', lowerbound=-1, upperbound=2, title="Ex3 Motif 4: Information-JSD, FlipRowFalse", flip_rows=False)

# %%

    histogram_scores(ic_jsd)

  
# %%
    
    boot = scoring_bootstrap(ic_jsd, myseed=42, iterations=1000, threshold=1.2, direction='main')

    
    
# %%

    top_score = max(candidate["score"] for candidate in dia)

    # count proportion of values that are geq than observed top score 

    p_value = np.sum(np.array(boot) >= top_score) / len(boot)
    print(f"Computed p-value: {p_value}")

    histogram_scores(np.array(boot), title=f"Distribution of Bootstrapped Top Scores (p={round(p_value, 4)})", top_score=top_score)


# %%

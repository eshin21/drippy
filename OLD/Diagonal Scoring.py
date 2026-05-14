from Bio import motifs
from Bio import SeqIO 
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr



def _visualize_matrix(input_matrix, colorscheme, lowerbound, upperbound, title, flip_rows=False):

    display_matrix = np.array(input_matrix.copy(), dtype=float)
    num_positions = display_matrix.shape[0]
    
    # for reverse complement, we manually do a flip of one of the axes to force the diagonal runs to be in the same direction as the direct repeats. But we also have to fix the axis labels in the visualization to reflect the flip 
    if flip_rows:
        row_labels = list(range(num_positions - 1, -1, -1))
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




####################################################################################
## FILE I/O

# read_excel("ExcelData/RC_Meme-3-Motif-6_IC_JSD_reversed.xlsx")

ic_jsd = pd.read_excel("ExcelData/RC_Simple_IC_JSD_reversed.xlsx", header=0, index_col=0)

ic_jsd_unflipped = ic_jsd[::-1]## we shouldn't need to do this because if we flip it, then we have broken the axis of symmetry which means we have to write another if else statement to specify the right portion of the matrix to focus on.


ic_jsd_np = ic_jsd.to_numpy()
##########################################################
#Viz
####################################################################################

_visualize_matrix(ic_jsd_unflipped, colorscheme='viridis', lowerbound=-1, upperbound=2, title="New Metric: Information    - JSD", flip_rows=False)


#####################################################################################
# Diagonal scoring
#####################################################################################

# %% 

def score_diagonals(matrix, threshold, direction='main'):
    n = matrix.shape[0]
    all_candidates = []

    for start_i in range(n):
        for start_j in range(start_i): # strictly checks below main diagonal. The main diagonal axis of symmetry will be the same for both direct and RC matrices.

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
            
            # Save all candidate diagonals greater than length 2
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


ic_jsd_unflipped_np = ic_jsd_unflipped.to_numpy()

test = score_diagonals(ic_jsd_unflipped_np, threshold=1.5, direction='main')

# ic_jsd_unflipped.to_excel("unflipped_RC_simple_scoring_ic_jsd.xlsx")


# %%

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

# ic_jsd_flipped = ic_jsd[::-1]## we shouldn't need to do thi

ic_jsd_np = ic_jsd.to_numpy()


pd.DataFrame(ic_jsd_np)
##########################################################
#Viz
####################################################################################

_visualize_matrix(ic_jsd_np, colorscheme='viridis', lowerbound=-1, upperbound=2, title="New Metric: Information    - JSD", flip_rows=False)

#####################################################################################
# Diagonal scoring
#####################################################################################

"""
    Find the longest diagonal (or anti-diagonal) run in the lower triangle
    where all values >= threshold.
    Parameters
    ----------
    matrix     : 2D numpy array (symmetric)
    threshold  : float, minimum cell value to count as a hit
    direction  : 'main' for diagonals parallel to the main diagonal,
                 'anti' for anti-diagonals
    Returns
    -------
    list of (i, j) tuples for the longest qualifying diagonal run,
    expressed as (row, col) in the lower triangle
"""

def score_diagonals(matrix, threshold, direction='main'):
    n = matrix.shape[0]
    best_len = 0
    best_coords = []
    for start_i in range(n):
        for start_j in range(n):     
            if direction == 'main' and (start_i - start_j) < 1: #only check below main diagonal 
                continue #skip to next iteration +j . Will only analyze region where row i (y-axis) > column j (x-axis)
            if direction == 'anti' and (start_i - start_j) < 0: #only check including and below main diagonal
                continue # skip to next iteration +j. Will only analyze region where row i (y-axis) >= column j (x-axis)
            i = start_i
            j = start_j
            coords = []
            # this part will extend using the scoring threshold 
            while 0 <= i < n and 0 <= j < n:
                if matrix[i, j] >= threshold:
                    coords.append((i, j))
                    i += 1
                    j += 1 #if direction == 'main' else -1   
                else:
                    break
            if len(coords) > 1 and len(coords) > best_len: #change this to append to save all candidates ? 
                best_len = len(coords)
                best_coords = coords
    return best_coords

ic_jsd_np = ic_jsd.to_numpy()

score_diagonals(ic_jsd_np, threshold=1.5, direction='anti')

ic_jsd.to_excel("scoring_ic_jsd.xlsx")
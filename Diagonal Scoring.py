from Bio import motifs
from Bio import SeqIO 
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr




def _visualize_matrix(input_matrix, colorscheme, lowerbound, upperbound, title):

    display_matrix = np.array(input_matrix.copy(), dtype=float)
    np.fill_diagonal(display_matrix, np.nan)

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        display_matrix, 
        annot=True,       # Turn on if you want to see the numbers
        annot_kws={"size":7},
        fmt='.2f',
        cmap=colorscheme,
        vmin=lowerbound, vmax=upperbound,    
        square=True
    )
    ax.set_facecolor('white')    #NA cells white 
    plt.title(title)
    plt.xlabel("Motif Position")
    plt.ylabel("Motif Position")
    plt.show()


####################################################################################
## FILE I/O

# read_excel("ExcelData/RC_Meme-3-Motif-6_IC_JSD_reversed.xlsx")


####################################################################################
# New Metric
## Information-Distance : Information content -  JSD 
## bitwise information is penalized for divergence or distance as measured by JSD
####################################################################################



####################################################################################
#Viz
####################################################################################

_visualize_matrix(ic_jsd, colorscheme='viridis', lowerbound=-1, upperbound=2, title="New Metric: Information - JSD")

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
        for start_j in range(start_i):      # strict lower triangle: j < i
            if (start_i - start_j) < 1: #should we be checking identity diagonals for RCs?
                continue
            i = start_i
            j = start_j
            coords = []

            # this part will extend using the scoring threshold 
            while 0 <= i < n and 0 <= j < n:
                if matrix[i, j] >= threshold:
                    coords.append((i, j))
                    i += 1
                    j += 1 if direction == 'main' else -1
                else:
                    break
            if len(coords) > 1 and len(coords) > best_len: #change this to append to save all candidates ? 
                best_len = len(coords)
                best_coords = coords
    return best_coords

ic_jsd_np = ic_jsd.to_numpy()

score_diagonals(ic_jsd_np, threshold=1.5, direction='main')

ic_jsd.to_excel("scoring_ic_jsd.xlsx")
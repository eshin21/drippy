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
# FILE I/O
####################################################################################

meme_file = "IMPORTS/meme_out_3/meme.xml"


### Accessing direct sequences
with open(meme_file) as handle:
    motifsM = motifs.parse(handle, "meme")


i = 5 ## TTCC...GGAA


# i = 3 ## TGCAC...GTGCA

seq_in_motif = motifsM[i].alignment.sequences #contains all the sequences aligned -- save this to make fasta

seq_in_motif[i]
motif = (motifsM)[i]

aligned_seq_matrix = []
for i in motif.alignment.sequences:
    print(str(i))
    aligned_seq_matrix.append(list(str(i)))

df_seq = pd.DataFrame(aligned_seq_matrix)

df_seq

####################################################################################

# FILE I/O
fasta_file = "IMPORTS/reverse_simple_motif.fasta"

matrix = []
for record in SeqIO.parse(fasta_file, "fasta"):
    # print(record.seq)
    segment = list(record.seq)
    matrix.append(segment)

df_seq = pd.DataFrame(matrix)


# TACTG...CAGTA reverse complement
# Position:   1  2  3  4  5   6-12 (spacer)   13 14 15 16 17
# Left arm:   T  A  C  T  G   [degenerate]     C  A  G  T  A
#                                             ↑ exact RC of TACTG


####################################################################################
## Tally the bases.
####################################################################################
df_seq.shape
len(df_seq)
print(df_seq)

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


####################################################################################
# Reverse complementing 

## original row order: [A, C, G, T] = indices [0, 1, 2, 3]
# complement row order: [T, G, C, A] = indices [3, 2, 1, 0]
complement_ppm = ppm_np[[3, 2, 1, 0], :]

# reverse
length = ppm_np.shape[1]
reversed_indices = np.arange(length - 1, -1, -1,)
# start = L-1     begin at the last valid index
# stop  = -1     stop before reaching -1 (so last value included is 0)
# step  = -1     decrement by 1 each time

rc_ppm = complement_ppm[:, reversed_indices]


# check : row 0 of complement should equal row 3 of original (A->T)
np.allclose(complement_ppm[0, :], ppm_np[3, :])  # T row

# check : first column of rc should equal last column of complement
np.allclose(rc_ppm[:, 0], complement_ppm[:, length-1])


# H_before = uniform 2 bits
# H_after = average the probabilities of bot


####################################################################################
# Positional Informational Content
# H_before = uniform 2 bits
# H_after = average the probabilities of both columns, calculate entropy from there 
####################################################################################

# H_before calcluation (so user can set it)
bg_probs_dict = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
bg_probs = np.array([bg_probs_dict[b] for b in ['A', 'C', 'G', 'T']])

H_before = 0 

for i in range(len(bg_probs)):
    p = bg_probs[i]
    term = p * np.log2(p)
    H_before = H_before + term

H_before = -H_before
print(H_before)

## calculations 
# storage
num_positions = ppm_np.shape[1]
after_df = pd.DataFrame(0.0, index=range(num_positions), columns=range(num_positions))

for i in range(num_positions):
    x = ppm_np[:, i]  

    for j in range(num_positions):

        y = complement_ppm[:, j]  ## key difference here -- we have to use the complement matrix. We compare reverse from an outwwards-in fashion, unlike the direct repeat where we do right to left pairs
        # y = complement_ppm[:, num_positions-1-j]  ## key difference here -- we have to use the complement matrix. We compare reverse from an outwwards-in fashion, unlike the direct repeat where we do right to left pairs
        

        xy = (x + y) / 2 
        H_after = -sum(xy * np.log2(xy + 1e-10)) #add a pseudocount because some values a0, log0 NaN
        after_df.iloc[i, j] = H_after

info_content_res = H_before - after_df




####################################################################################
#IC Viz
####################################################################################

_visualize_matrix(info_content_res, colorscheme='viridis_r', lowerbound=0, upperbound=2, title="Information")

_visualize_matrix(info_content_res[::-1], colorscheme='viridis_r', lowerbound=0, upperbound=2, title="Information, flipped", flip_rows=True)




####################################################################################
##### Metric: Pearson
####################################################################################

pearson_results_df = pd.DataFrame(0.0, index=range(num_positions), columns=range(num_positions))

for i in range(ppm.shape[1]):
    x = ppm_np[:, i]

    for j in range(ppm.shape[1]):
        y = complement_ppm[:, j]
        pearson_results_df.iloc[i, j] = pearsonr(x, y)[0]

_visualize_matrix(input_matrix = pearson_results_df, colorscheme='viridis', lowerbound=0, upperbound=1, title="Pearson")




####################################################################################
##### Metric: Jensen–Shannon divergence
####################################################################################
num_positions = ppm.shape[1]
jsd_results_df = pd.DataFrame(0.0, index=range(num_positions), columns=range(num_positions))

for i in range(ppm.shape[1]):
    x = ppm_np[:, i] + 1e-10

    for j in range(ppm.shape[1]):
    
        y = complement_ppm[:, j] + 1e-10
        midpoint = (x + y) / 2 
        D_XM = sum(x * np.log2(x / midpoint))
        D_YM = sum(y * np.log2(y / midpoint))
        JSD = 0.5 * (D_XM + D_YM)
        jsd_results_df.iloc[i, j] = JSD




input_matrix = jsd_results_df.copy()

_visualize_matrix(jsd_results_df, colorscheme='viridis_r', lowerbound=0, upperbound=1, title="Metric: Jensen Shannon")


#######################################



####################################################################################
# New Metric
## Information-Distance : Information content -  JSD 
## bitwise information is penalized for divergence or distance as measured by JSD
####################################################################################

ic_jsd = info_content_res - jsd_results_df


####################################################################################
#Viz
#######################################ic#############################################

_visualize_matrix(ic_jsd, colorscheme='viridis', lowerbound=-1, upperbound=2, title="New Metric: Information - JSD")



_visualize_matrix(ic_jsd[::-1], colorscheme='viridis', lowerbound=-1, upperbound=2, title="New Metric: Information - JSD, flipped")



ic_jsd[::-1].to_excel("ExcelData/RC_Meme-3-Motif-6_IC_JSD_reversed.xlsx")


ic_jsd.index

####################################################################################
# New Metric
## Correlation-Scaled Positional Information Content
## potential problem with signs of correlation.... see [0, 9] with toy example 
####################################################################################

ic_corr = info_content_res * pearson_results_df

_visualize_matrix(ic_corr, 'viridis', lowerbound=-2, upperbound=2, title="New Metric: Information * Correlation")

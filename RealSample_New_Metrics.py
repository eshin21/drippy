
from Bio import motifs
from Bio import SeqIO 
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr




def _visualize_matrix(input_matrix, lowerbound, upperbound, title):

    display_matrix = np.array(input_matrix.copy(), dtype=float)
    np.fill_diagonal(display_matrix, np.nan)

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        display_matrix, 
        annot=True,       # Turn on if you want to see the numbers
        annot_kws={"size":8},
        fmt='.2f',
        cmap='viridis',
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


meme_file = "meme_out_2/meme.xml"


### Accessing direct sequences
with open(meme_file) as handle:
    motifsM = motifs.parse(handle, "meme")


i = 5
seq_in_motif = motifsM[i].alignment.sequences #contains all the sequences aligned -- save this to make fasta

# obvious example  TTAA....TTAA
motif = (motifsM)[i]

aligned_seq_matrix = []
for i in motif.alignment.sequences:
    print(str(i))
    aligned_seq_matrix.append(list(str(i)))

df_seq = pd.DataFrame(aligned_seq_matrix)


####################################################################################


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
num_positions = ppm.shape[1]
after_df = pd.DataFrame(0.0, index=range(num_positions), columns=range(num_positions))

for i in range(ppm.shape[1]):
    x = ppm.iloc[:, i]

    for j in range(ppm.shape[1]):
        if(i == j): 
            after_df.iloc[i, j] = 0 
            continue # we dont need to do identity
        else:
            y = ppm.iloc[:, j]
            xy = (x + y) / 2 
            H_after = -sum(xy * np.log2(xy + 1e-10)) #add a pseudocount because some values are 0, log0 NaN
            after_df.iloc[i, j] = H_after

info_content_res = H_before - after_df


####################################################################################
#IC Viz
####################################################################################


input_matrix = info_content_res.copy()

plt.figure(figsize=(10, 8))
ax = sns.heatmap(
    input_matrix, 
    annot=True,       # Turn on if you want to see the numbers
    annot_kws={"size":8},
    cmap='viridis',   
    vmin=0, vmax=2,    
    square=True
)


plt.title("Information")
plt.xlabel("Motif Position")
plt.ylabel("Motif Position")
plt.show()

# _visualize_matrix(input_matrix, lowerbound=0, upperbound=2, title="hello")



####################################################################################
##### Metric: Jensen–Shannon divergence
####################################################################################
num_positions = ppm.shape[1]
jsd_results_df = pd.DataFrame(0.0, index=range(num_positions), columns=range(num_positions))

for i in range(ppm.shape[1]):
    x = ppm.iloc[:, i] + 1e-10

    for j in range(ppm.shape[1]):
        if(i == j): 
            jsd_results_df.iloc[i, j] = 0 
            continue # we dont need to do identity
        else:
            y = ppm.iloc[:, j] + 1e-10
            midpoint = (x + y) / 2 
            D_XM = sum(x * np.log2(x / midpoint))
            D_YM = sum(y * np.log2(y / midpoint))
            JSD = 0.5 * (D_XM + D_YM)
            jsd_results_df.iloc[i, j] = JSD




input_matrix = jsd_results_df.copy()

_visualize_matrix(jsd_results_df, lowerbound=0, upperbound=1, title="Metric: Jensen Shannon")


####################################################################################
##### Metric: Pearson
####################################################################################

pearson_results_df = pd.DataFrame(0.0, index=range(num_positions), columns=range(num_positions))

for i in range(ppm.shape[1]):
    x = ppm.iloc[:, i]

    for j in range(ppm.shape[1]):
        if(i == j): 
            pearson_results_df.iloc[i, j] = 0 
            continue # we dont need to do identity
        else:
            y = ppm.iloc[:, j]
           
            pearson_results_df.iloc[i, j] = pearsonr(x, y)[0]



_visualize_matrix(input_matrix = pearson_results_df, lowerbound=0, upperbound=1, title="Pearson")



####################################################################################
# New Metric
## Information-Distance : Information content -  JSD 
## bitwise information is penalized for divergence or distance as measured by JSD
####################################################################################

ic_jsd = info_content_res - jsd_results_df


####################################################################################
#Viz
####################################################################################

_visualize_matrix(ic_jsd, lowerbound=-1, upperbound=2, title="New Metric: Information - JSD")



####################################################################################
# New Metric
## Correlation-Scaled Positional Information Content
## potential problem with signs of correlation.... see [0, 9] with toy example 
####################################################################################

ic_corr = info_content_res * pearson_results_df

_visualize_matrix(ic_corr, lowerbound=-2, upperbound=2, title="New Metric: Information * Correlation")

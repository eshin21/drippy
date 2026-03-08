
from Bio import motifs
from Bio import SeqIO 
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


####################################################################################
## FILE I/O
fasta_file = "simple_motif.fasta"

matrix = []
for record in SeqIO.parse(fasta_file, "fasta"):
    # print(record.seq)
    segment = list(record.seq)
    matrix.append(segment)

df = pd.DataFrame(matrix)
####################################################################################


####################################################################################
## Tally the bases.
####################################################################################
df.shape
len(df)
print(df.shape[1])

# make running storage for each base for each column index
# our new data frame will have 12 columns and 4 rows 
A_tally = []
C_tally = []
G_tally = []
T_tally = []

for column_index in range(df.shape[1]): 
    current_column = list(df[column_index])

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

# flip transpose 

psfm = psfm.transpose()

# probability matrix (counts in each column divided by number of sequences)

ppm = pd.DataFrame(psfm / len(df))


####################################################################################
##compare to biopython xml read 
####################################################################################
meme_file = "meme_out_simple/meme.xml"
with open(meme_file) as f:
    record = motifs.parse(f, "meme")
print(f"N = {len(record)} motifs in this file.\n")
motif = (record)[0]

test = pd.DataFrame(motif.pwm)
test = test.transpose()
test.equals(ppm) #slight differences, investigate later
# ppm.to_excel("ppm.xlsx")

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

IC = H_before - after_df
input_matrix = IC.copy()


####################################################################################
#Viz
####################################################################################

plt.figure(figsize=(10, 8))
# Create labels starting from 1 up to the motif length
labels = np.arange(1, motif.length + 1)

ax = sns.heatmap(
    input_matrix, 
    annot=False,       # Turn on if you want to see the numbers
    cmap='viridis_r',    # Viridis is good for magnitude (0 to 2)
    vmin=0, vmax=2,    # JSD  ranges from 0 to 1 bit
    square=True,
    xticklabels=labels,
    yticklabels=labels
)
# Set the background color to white so that NaN values (diagonal and filtered) appear white
ax.set_facecolor('white')

plt.title("Simple Example - IC")
plt.xlabel("Motif Position")
plt.ylabel("Motif Position")
plt.show()

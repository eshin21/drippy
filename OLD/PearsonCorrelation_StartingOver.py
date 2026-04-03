
from Bio import motifs
from Bio import SeqIO 
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


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
# Pearson correlation

####################################################################################

## calculations 
# storage
num_positions = ppm.shape[1]
results_df = pd.DataFrame(0.0, index=range(num_positions), columns=range(num_positions))

for i in range(ppm.shape[1]):
    x = ppm.iloc[:, i]

    for j in range(ppm.shape[1]):
        if(i == j): 
            results_df.iloc[i, j] = 0 
            continue # we dont need to do identity
        else:
            y = ppm.iloc[:, j]
           
            results_df.iloc[i, j] = pearsonr(x, y)[0]


input_matrix = results_df.copy()

input_matrix.to_excel("corr_res.xlsx")



####################################################################################
#Viz
####################################################################################

plt.figure(figsize=(10, 8))

ax = sns.heatmap(
    input_matrix, 
    annot=True,       
    cmap='viridis_r',   
    vmin=-1, vmax=1,    
    square=True,

)

plt.title("Simple Example - Correlation")
plt.xlabel("Motif Position")
plt.ylabel("Motif Position")
plt.show()

### HEADER

## Author: Enoch Shin
## Purpose: This file aims to ingest MEME XML files  (Multiple Em for Motif Elicitation) to detect direct repeats or inverted repeats (reverse complements) in the list of detected motifs.

## Method: using the PSSM matrix (or other matrices if needed), perform self-correlation of the motif to detect direct repeats and inverted repeats (diagonal runs with high correlation, or some other appropriate metric such as information content) 

from Bio import motifs
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


## FILE I/O
meme_file = "meme_out_1/meme.xml"



### Accessing direct sequences
with open(meme_file) as handle:
    motifsM = motifs.parse(handle, "meme")
all_seq = motifsM[0].alignment.sequences
all_seq[1] ## corresponds to HTML page's More --> first sequence that appears in list 


with open(meme_file) as f:
    record = motifs.parse(f, "meme")

print(f"N = {len(record)} motifs in this file.\n")


#grab the first one

i = 0
motif = (record)[0]
    

print(f"Consensus Sequence: {motif.consensus}")
print(f"Length: {motif.length} bp")
print(f"E-value: {motif.evalue}") # Useful for filtering noise later [1]
    
    
motif.background

### IMPORTANT -- there are some columns for which a base doesn't appear at all, so the score log2 (Observed Probabilty / Background Probability) hits -Inf when the denominator is zero
# therefore we will use biopython's motif pseudocounts feature 
# allegedly a publication standard way of doing things.
#  
motif.pseudocounts = 1.0

pssm = motif.pssm #; pssm


# 5. Convert to a NumPy Array for  "Sliding" Algorithm
# # We explicitly order rows as A, C, G, T to ensure consistent math later.
# Shape will be (4, Length)
matrix_data = np.array([
    pssm['A'],
    pssm['C'],
    pssm['G'],
    pssm['T']
])


    
matrix_data


print(f"Matrix shape (Bases x Length): {matrix_data.shape}")
print("PSSM Scores at Position 1 (A, C, G, T):")
print(pd.Series(matrix_data[:, 0], index=['A', 'C', 'G', 'T']))
print("\n")


########    DIRECT REPEATS

correlation_matrix = np.corrcoef(matrix_data, rowvar=False)

# region Investigating Negative Correlations

#### checking motif locations for negative correlations 
## these won't directly show inverted repeats, but could be useful

# Get off-diagonal correlations (lower triangle to avoid duplicates)
rows, cols = np.tril_indices_from(correlation_matrix, k=-1)
corrs = correlation_matrix[rows, cols]

# Find indices of the 10 smallest (most negative) correlations
num_to_show = 10
sorted_indices = np.argsort(corrs)[:num_to_show]

print(f"Bottom {num_to_show} correlations:")
for idx in sorted_indices:
    r, c = rows[idx], cols[idx]
    # Ensure i < j for display
    i, j = (c, r) if c < r else (r, c)
    val = corrs[idx]
    
    print(f"Correlation: {val:.4f}")
    print(f"Indices (0-based): [{i} {j}]")
    print(f"Indices (1-based): [{i+1} {j+1}]")

    print(f"PSSM Scores for Position {i+1} vs Position {j+1}:")
    print(pd.DataFrame({
        f'Pos {i+1}': matrix_data[:, i],
        f'Pos {j+1}': matrix_data[:, j]
    }, index=['A', 'C', 'G', 'T']))
    print("-" * 20)

# endregion

# region Investigating ties in correlations
## show PSSM at motif indices that have ties in PSSM values
print("Checking for ties in PSSM scores (ambiguous consensus):")
for i in range(matrix_data.shape[1]):
    col = matrix_data[:, i]
    max_val = np.max(col)
    # Check for ties in the maximum score
    if np.sum(np.isclose(col, max_val)) > 1:
        print(f"Position {i+1} has a tie for the top score:")
        print(pd.Series(col, index=['A', 'C', 'G', 'T']))
        print("-" * 20)
    else: 
        print("no ties")
# endregion


## Manual blankout heuristics:




# 1. Filter by consensus base identity
# Get the index of the max score for each position (0=A, 1=C, 2=G, 3=T)
consensus_indices = np.argmax(matrix_data, axis=0)
# Create a boolean matrix where True means both positions have the same consensus base
base_match_mask = (consensus_indices[:, None] == consensus_indices[None, :])
# Apply mask: set correlations to NaN where bases don't match
correlation_matrix[~base_match_mask] = np.nan

# 1.5 Remove diagonals for which the pairwise distance between the motif indices is <3 bp (to focus on longer repeats)
x, y = np.indices(correlation_matrix.shape)
correlation_matrix[np.abs(x - y) < 3] = np.nan

# 1.75 Blank out values where the continuous diagonal of positive correlations is less than 3 cells long
valid = ~np.isnan(correlation_matrix)
# Ensure positive correlation (though base matching usually implies this, we enforce it as per comment)
valid &= (np.nan_to_num(correlation_matrix, nan=-1.0) > 0)

# Create shifted views to check neighbors along the diagonal
# Down-right shifts (looking at previous elements)
prev1 = np.zeros_like(valid)
prev1[1:, 1:] = valid[:-1, :-1]
prev2 = np.zeros_like(valid)
prev2[2:, 2:] = valid[:-2, :-2]

# Up-left shifts (looking at next elements)
next1 = np.zeros_like(valid)
next1[:-1, :-1] = valid[1:, 1:]
next2 = np.zeros_like(valid)
next2[:-2, :-2] = valid[2:, 2:]

# Keep if part of a run >= 3 (start, middle, or end)
keep_mask = valid & ((prev1 & prev2) | (next1 & next2) | (prev1 & next1))
correlation_matrix[~keep_mask] = np.nan

# 2. Blank out correlations < 0.5 (weak correlation)
correlation_matrix[correlation_matrix < 0.5] = np.nan

# 3. Identity correlation will always be 1.0, so we can blank out the diagonal to focus on off-diagonal correlations 
np.fill_diagonal(correlation_matrix, -1.0)

correlation_matrix_narrowed = correlation_matrix.copy()




# Step 2: Plot the Heatmap  
plt.figure(figsize=(10, 8))
# Create labels starting from 1 up to the motif length
labels = np.arange(1, motif.length + 1)

sns.heatmap(
    correlation_matrix_narrowed, 
    annot=False,       # Turn on if you want to see the numbers
    cmap='coolwarm',   # Red = Positive correlation, Blue = Negative
    vmin=-1, vmax=1,   # Fix scale from -1 to 1 for consistency
    square=True,
    xticklabels=labels,
    yticklabels=labels
)

plt.title("Example 1 - PSSM Self-Correlation (Direct Repeats) \n>0.5 Correlation Only \nRemove Non-Matching Bases \n Remove <3 bp Diagonal Runs")
plt.xlabel("Motif Position")
plt.ylabel("Motif Position")
plt.show()





# At the moment we are only concerned with direct repeats, so let's blank out anything with 

# eliminate <3 bp diagonal runs
# 

correlation_matrix

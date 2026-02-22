### HEADER

## Author: Enoch Shin
## Purpose: This file aims to ingest MEME XML files  (Multiple Em for Motif Elicitation) to detect direct repeats or inverted repeats (reverse complements) in the list of detected motifs.

## in this file, we will use Info Content -- can try two ways
# KL divergence -- how much extra info do we need to represent distribution B using distribution A?
# Mutual information: if we know the nucleotide distribution of Column X, does that reduce the entropy (uncertainty) of Column Y?

# KL Divergence 
# key properties: Nonnegative. Asymmetric (comparing A to B is diff from B to A). Equals 0 when the distributions are the same.
# so we would look for close-to-zero metrics for KL divergence
# null assumption -- both column A and B will be identical in   their probability distributions (PSFM).
# ^key vulnerability of the KL divergence -- noise. If Column X's PSFM for A, C, T, and G is 0.80, 0.10, 0.00, 0.10, versus Column Y's PSFM being 0.80, 0.00, 0.10, 0.10, the KL divergence may fail to recognize that this is actually a direct repeat, because the distributions "look" different across all 4bp.

# "positional information content" -- suggested by Ivan 
# Mutual Information // Drop in Entropy 
# For each pairwise index in the PSFM, average column X and Y's cellwise PSFM values. Then with this new matrix, compute Shannon's Entropy. Then, using the background entropy from the available Biopython package, find the difference (i.e. drop in entropy). 

# This will dilute out the noise of the minority probabilities while preserving signal for the mutual majority class.

from Bio import motifs
from Bio.Seq import Seq
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

####################################################################################
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

# Grab the first one
i = 0
motif = (record)[1]
####################################################################################



####################################################################################
# Calculate background entropy
# background is used for baseline noise measure to calculate drop in noise
# bg_probs = np.array([motif.background[b] for b in ['A', 'C', 'G', 'T']])
motif.background
#  noticed that the background is always 0.25 uniform for whatever reason. 
# Find this from MEME HTML file > Inputs & Settings 


bg_probs_dict = {"A": 0.311, "C": 0.189, "G": 0.189, "T": 0.311}
bg_probs = np.array([bg_probs_dict[b] for b in ['A', 'C', 'G', 'T']])
####################################################################################

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

print(f"Matrix shape (Bases x Length): {matrix_data.shape}")
print("PSSM Scores at Position 1 (A, C, G, T):")
print(pd.Series(matrix_data[:, 0], index=['A', 'C', 'G', 'T']))
print("\n")


####################################################################################
########    DIRECT REPEATS
####################################################################################

# 1. Average the pairwise column probabilities.
# We need probabilities (PWM) for entropy calculations, not log-odds (PSSM).
# confusingly, Many researchers use the term "PWM" to refer to the log-odds scoring matrix, but in Biopython, pwm specifically refers to the probability matrix, while pssm refers to the scoring matrix 

pwm = motif.pwm

pwm_data = np.array([pwm['A'], pwm['C'], pwm['G'], pwm['T']])

#Calculate background entropy with shannon's formula 
bg_entropy = -np.sum(bg_probs * np.log2(bg_probs))


# Pairwise average of columns: (P_i + P_j) / 2
# Broadcasting: (4, L, 1) + (4, 1, L) -> (4, L, L)
pairwise_avg = 0.5 * (pwm_data[:, :, None] + pwm_data[:, None, :])

# 2. Compute Shannon's entropy pairwise.
# H = -sum(p * log2(p))
pairwise_entropy = -np.sum(pairwise_avg * np.log2(pairwise_avg + 1e-15), axis=0)

# 3. Compute the difference between the background entropy and the computed Shannon's entropy for each pairwise column.
info_matrix = bg_entropy - pairwise_entropy

# Use info_matrix for downstream visualization
input_matrix = info_matrix.copy()


# 4. Plot the matrix.

####################################################################################
# region Investigating Top Information Content Pairs

#### Checking motif locations for highest information content (strongest repeats)

# Get off-diagonal (lower triangle to avoid duplicates)
rows, cols = np.tril_indices_from(input_matrix, k=-1)
vals = input_matrix[rows, cols]

# Find indices of the 10 largest values (highest information content)
num_to_show = 10
sorted_indices = np.argsort(vals)[-num_to_show:][::-1]

print(f"Top {num_to_show} Information Content Pairs:")
for idx in sorted_indices:
    r, c = rows[idx], cols[idx]
    # Ensure i < j for display
    i, j = (c, r) if c < r else (r, c)
    val = vals[idx]
    
    print(f"Info Content: {val:.4f} bits")
    print(f"Indices (0-based): [{i} {j}]")
    print(f"Indices (1-based): [{i+1} {j+1}]")

    print(f"PSSM Scores for Position {i+1} vs Position {j+1}:")
    print(pd.DataFrame({
        f'Pos {i+1}': matrix_data[:, i],
        f'Pos {j+1}': matrix_data[:, j]
    }, index=['A', 'C', 'G', 'T']))
    print("-" * 20)

# endregion
####################################################################################

####################################################################################
# region Investigating ties
####################################################################################
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
####################################################################################


####################################################################################
#region Manual blankout heuristics:
# 1. Filter by consensus base identity
# Get the index of the max score for each position (0=A, 1=C, 2=G, 3=T)
consensus_indices = np.argmax(matrix_data, axis=0)
# Create a boolean matrix where True means both positions have the same consensus base
base_match_mask = (consensus_indices[:, None] == consensus_indices[None, :])
# Apply mask: set values to NaN where bases don't match
input_matrix[~base_match_mask] = np.nan

# 1.5 Remove diagonals for which the pairwise distance between the motif indices is <3 bp (to focus on "repeats with encoding stuff in between")
x, y = np.indices(input_matrix.shape)
input_matrix[np.abs(x - y) < 3] = np.nan

# 1.75 Blank out values where the continuous diagonal of positive correlations is less than 3 cells long
valid = ~np.isnan(input_matrix)
# Ensure positive correlation (though base matching usually implies this, we enforce it as per comment)
valid &= (np.nan_to_num(input_matrix, nan=-1.0) > 0)

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
input_matrix[~keep_mask] = np.nan

# # 2. Blank out correlations < 0.5 (weak correlation)
# input_matrix[input_matrix < 0.5] = np.nan

# # 3. Identity correlation will always be 1.0, so we can blank out the diagonal to focus on off-diagonal correlations 
# np.fill_diagonal(input_matrix, np.nan)

input_matrix = input_matrix.copy()

#endregion 





####################################################################################
# Step 2: Plot the Heatmap 
# 
#  
plt.figure(figsize=(10, 8))
# Create labels starting from 1 up to the motif length
labels = np.arange(1, motif.length + 1)

ax = sns.heatmap(
    input_matrix, 
    annot=False,       # Turn on if you want to see the numbers
    cmap='viridis',    # Viridis is good for magnitude (0 to 2)
    vmin=0, vmax=2,    # Information content ranges from 0 to 2 bits
    square=True,
    xticklabels=labels,
    yticklabels=labels
)
# Set the background color to white so that NaN values (diagonal and filtered) appear white
ax.set_facecolor('white')

plt.title("Example 1 - Pairwise Positional Information Content (Direct Repeats), \n Pairwise Consensus Bases \n >3 distance b/w indices \n >3 diagonal runs only")
plt.xlabel("Motif Position")
plt.ylabel("Motif Position")
plt.show()

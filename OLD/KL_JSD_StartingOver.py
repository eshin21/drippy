
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
# Kullback–Leibler divergence
####################################################################################
## calculations 
# storage
num_positions = ppm.shape[1]
results_df = pd.DataFrame(0.0, index=range(num_positions), columns=range(num_positions))


for i in range(num_positions):
    for j in range(num_positions):
        
        if i == j:
            results_df.iloc[i, j] = 0 
            continue
            
        # Extract the two columns from PPM
        P = ppm.iloc[:, j]
        Q = ppm.iloc[:, i] 

            
        # Calculate KL Divergence
        # Add pseudocount

        kl_div = sum(P * np.log2((P + 1e-10) / (Q + 1e-10)))
        results_df.iloc[i, j] = kl_div

print(results_df)

results_df.to_excel("res.xlsx")


input_matrix = results_df.copy()


####################################################################################
#Viz
####################################################################################

plt.figure(figsize=(10, 8))

# - annot=True: Adds the numbers inside the boxes
# - fmt=".2f": Formats the numbers to 2 decimal places (like your IC matrix)
# - cmap="viridis_r": Uses a color map similar to your IC matrix ('viridis' is common, '_r' reverses it so dark is high, light is low)
# - cbar_kws: Adds a label to the color bar
ax = sns.heatmap(results_df, 
                 annot=True, 
                 fmt=".1f", 
                 cmap="viridis", 
                 cbar_kws={'label': 'KL Divergence'}
                 )

# 3. Add labels and title
plt.title('KL Divergence Matrix', fontsize=16)
plt.xlabel('Denominator Column (Q)', fontsize=12)
plt.ylabel('Numerator Column (P)', fontsize=12)

# 4. Optional: Force the x and y axis ticks to be integers if they aren't already
# ax.set_xticks(np.arange(len(results_df.columns)) + 0.5)
# ax.set_xticklabels(results_df.columns)
# ax.set_yticks(np.arange(len(results_df.index)) + 0.5)
# ax.set_yticklabels(results_df.index)

# 5. Display the plot
plt.show()

# Optional: Save the figure
# plt.savefig("kl_divergence_heatmap.png", dpi=300, bbox_inches='tight')



###########################################
##### JSD
############################################

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

jsd_results_df.to_excel("jsd_res.xlsx")


input_matrix = jsd_results_df.copy()



plt.figure(figsize=(10, 8))


ax = sns.heatmap(
    input_matrix, 
    annot=True,       # Turn on if you want to see the numbers
    cmap='viridis',   
    vmin=0, vmax=1,    
    square=True
)


plt.title("Simple Example - JSD")
plt.xlabel("Motif Position")
plt.ylabel("Motif Position")
plt.show()

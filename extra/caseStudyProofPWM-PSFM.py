
from Bio import motifs
from Bio.Seq import Seq
import numpy as np
import os
import pandas as pd

# Define instances and create motif
instances = [Seq("GAGG"), Seq("GACG"), Seq("GAGG"), Seq("TAGC")]
m = motifs.create(instances)
total_seqs = len(instances)

# 1. Manual PSFM Calculation
manual_psfm = {base: [] for base in "ACGT"}
for i in range(m.length):
    for base in "ACGT":
        freq = m.counts[base][i] / total_seqs
        manual_psfm[base].append(freq)

print("--- Manual PSFM ---")
for base in "ACGT":
    # Rounding for clean display
    formatted_freqs = [round(f, 2) for f in manual_psfm[base]]
    print(f"{base}: {formatted_freqs}")
    
m.pwm
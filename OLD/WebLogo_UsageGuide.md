## Method 1: Using Biopython's WrapperBiopython provides an intuitive wrapper for WebLogo. You can create a motif object from sequence instances and directly generate the logo.pythonfrom Bio import motifs

### 1. Create a motif object from a list of instances


```
from Bio.Seq import Seq
instances = [Seq("AGCT"), Seq("TCGA"), Seq("AACT")]
motif = motifs.create(instances)

```

### 2. Generate the weblogo (defaults to pdf, png, or eps)

```
# Kwds are passed directly to the weblogo generator
motif.weblogo("motif_logo.png", format="png")

```


## Method 2: Using the Native WebLogo 3 APIFor full granular control over titles, colors, and formatting options (without relying on Biopython classes), use WebLogo 3's native module.

```
from weblogo import *
````


### 1. Open and read your sequence data (FASTA format)

```
with open('cap.fa', 'r') as fin:
    seqs = read_seq_data(fin)

```

### 2. Configure Logo Data

```
logodata = LogoData.from_seqs(seqs)
```

### 3. Configure Logo Options

```
options = LogoOptions()
options.title = "Consensus Motif Logo"
options.color_scheme = std_ nucleic_acid # or std_protein for amino acids
```

### 4. Generate the format and output

```
format = LogoFormat(logodata, options)
with open('output.eps', 'wb') as fout:
    fout.write(eps_formatter(logodata, format))
````


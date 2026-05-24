To import functions from this file into a new Python script, you will first run into a minor naming issue.

Currently, your file is named Diagonal Scoring.py. Python's standard import statement does not support spaces in module names, so running import Diagonal Scoring will result in a SyntaxError.

Step 1: Rename your file
To make the file importable, I highly recommend renaming it to use an underscore instead of a space. For example, rename it to diagonal_scoring.py.

Step 2: Import into your new file
Once renamed, you can create a new Python file in the exact same directory (e.g., analysis.py) and import your functions using one of the standard Python import methods:

Option A: Import specific functions (Recommended) This allows you to call the functions directly by their name.

# analysis.py

# Import just the functions you need
from diagonal_scoring import score_diagonals, _visualize_matrix

import numpy as np

# Now you can use them directly
my_matrix = np.array([[1, 0], [0, 1]])
candidates = score_diagonals(my_matrix, threshold=1.5, direction='main')

Option B: Import the entire module This imports the whole file, and you access the functions using dot notation.

# analysis.py

import diagonal_scoring
import numpy as np

my_matrix = np.array([[1, 0], [0, 1]])

# Use dot notation to call the function
candidates = diagonal_scoring.score_diagonals(my_matrix, threshold=1.5)

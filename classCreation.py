########################################################################
# Diagonal scoring based on threshold
## using a class to maintain threshold and direction state
########################################################################

def shuffle_metrics(metrics_matrix, myseed = 42):
    
    # 1. Set the seed by creating a random number generator (rng)
    rng = np.random.default_rng(seed=myseed)

    # 2. Get the number of positions (assuming a square matrix)
    n = metrics_matrix.shape[0]
    
    # 3. Create a random permutation of the indices
    indices = rng.permutation(n)

    # 4. Apply the same shuffled indices to both rows and columns
    shuffled_matrix = metrics_matrix[np.ix_(indices, indices)]

    return shuffled_matrix


##################################################################
## bootstrap construction
##################################################################

class DiagonalScorer:
    def __init__(self, threshold=1.0, direction='main'):
        self.threshold = threshold
        self.direction = direction

    def score(self, matrix):
        n = matrix.shape[0]
        all_candidates = []

        for start_i in range(n):
            for start_j in range(start_i): # check below main diagonal.
                i, j = start_i, start_j
                coords = []
                score_sum = 0

                while 0 <= i < n and 0 <= j < n:
                    if matrix[i, j] >= self.threshold:
                        coords.append((i, j))
                        score_sum += matrix[i, j]
                        i += 1
                        j += 1 if self.direction == 'main' else -1   
                    else:
                        break
                
                # Save all diagonals greater or eq than length 2
                if len(coords) >= 2: 
                    all_candidates.append({
                        "coords": coords,
                        "length": len(coords),
                        "score": score_sum
                    })

        all_candidates.sort(key=lambda x: x["length"], reverse=True)
        
        filtered_candidates = []
        for candidate in all_candidates:
            c_set = set(candidate["coords"])
            is_subset = any(c_set.issubset(set(kept["coords"])) for kept in filtered_candidates)
            if not is_subset:
                filtered_candidates.append(candidate)

        return filtered_candidates

    def bootstrap(self, metrics_matrix, myseed=42, iterations=1000):
        bootstrap_scores = []

        for i in range(iterations):
            shuffled = shuffle_metrics(metrics_matrix, myseed=myseed + i)
            dia = self.score(shuffled)
            
            if dia:
                top_score = max(candidate["score"] for candidate in dia)
                bootstrap_scores.append(top_score)

        return bootstrap_scores
    
    ic_jsd = compute_metrics(ppm, metric='PIC-JSD', direction='reverse')

    scorer = DiagonalScorer(threshold=1.2, direction='reverse')
    dia = scorer.score(ic_jsd)
    pd.DataFrame(dia)

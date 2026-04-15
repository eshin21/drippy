def score_diagonals(matrix, threshold, direction='main'):
    n = matrix.shape[0]
    best_len = 0
    best_coords = []

    for start_i in range(n):
        for start_j in range(start_i):      # strict lower triangle: j < i
            if (start_i - start_j) < 1: #should we be checking identity diagonals for RCs?
                continue
            i = start_i
            j = start_j
            coords = []

            # this part will extend using the scoring threshold 
            while 0 <= i < n and 0 <= j < n:
                if matrix[i, j] >= threshold:
                    coords.append((i, j))
                    i += 1
                    j += 1 if direction == 'main' else -1   
                else:
                    break
            if len(coords) > 1 and len(coords) > best_len: #change this to append to save all candidates ? 
                best_len = len(coords)
                best_coords = coords
    return best_coords
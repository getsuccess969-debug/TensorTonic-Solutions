import numpy as np

def entropy_node(y):
    # 1. Use np.unique to get counts
    values, counts = np.unique(y, return_counts=True)
    
    # 2. Rename variable to 'n' to avoid shadowing built-in len()
    n = len(y) 
    
    # 3. Handle the edge case: entropy of an empty list is 0
    if n == 0:
        return 0.0
    
    # 4. Vectorized Probability calculation
    p = counts / n
    
    # 5. Vectorized Entropy: H = -sum(p * log2(p))
    # We use np.sum to replace the 'for' loop
    H = -np.sum(p * np.log2(p))
    
    return H
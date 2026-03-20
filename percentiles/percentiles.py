import numpy as np

def percentiles(x, q):
    x=np.sort(x)
    q=np.array(q)
    n=x.shape[0]
    L=np.percentile(x,q,method="linear")
    return L
    
    
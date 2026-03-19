import numpy as np

def expected_value_discrete(x, p):
    x,p=map(np.asarray,[x,p])
    E=x.dot(p)
    if (p.sum()!=1):
        raise ValueError("ValueError")
    else:
        return E

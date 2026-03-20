import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    w,g,s=map(np.array,[w,g,s])
    s=beta*s+(1-beta)*g*g
    w=w-lr/(np.sqrt(s+eps))*g
    return w,s
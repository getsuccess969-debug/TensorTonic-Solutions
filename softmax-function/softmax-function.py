import numpy as np

def softmax(x):
    x=np.array(x)
    if(len(x.shape)==1):
        return np.exp(x-np.max(x))/np.sum(np.exp(x-np.max(x)))
    else:
        return np.exp(x-np.max(x,axis=1,keepdims=True))/np.sum(np.exp(x-np.max(x,axis=1,keepdims=True)),axis=1,keepdims=True)
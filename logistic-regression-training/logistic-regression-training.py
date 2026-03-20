import numpy as np

def _sigmoid(z):
    # Clips z to avoid overflow in exp
    z = np.clip(z, -500, 500)
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def grad(y, p, X):
    n = len(y)
    # Use dot product for the gradient calculation
    # (Features^T) dot (Predictions - Targets)
    lw = (1/n) * np.dot(X.T, (p - y))
    lb = (1/n) * np.sum(p - y)
    return lw, lb

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    X = np.array(X)
    y = np.array(y)
    n_samples, n_features = X.shape
    
    # Initialize weights as a vector of zeros
    w = np.zeros(n_features)
    b = 0
    
    for i in range(steps):
        # Use matrix multiplication (@) for the linear step
        z = X @ w + b
        p = _sigmoid(z)
        
        lw, lb = grad(y, p, X)
        
        # Gradient Descent update
        w = w - lr * lw
        b = b - lr * lb

    return w, b
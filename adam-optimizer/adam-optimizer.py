import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    # Ensure inputs are numpy arrays for vectorization
    param, grad = np.asarray(param), np.asarray(grad)
    m, v = np.asarray(m), np.asarray(v)
    
    # 1. Update biased first moment estimate (Momentum)
    m = beta1 * m + (1 - beta1) * grad
    
    # 2. Update biased second raw moment estimate (Scaling)
    # Use **2 instead of ^
    v = beta2 * v + (1 - beta2) * (grad**2)
    
    # 3. Compute bias-corrected first moment estimate
    # Use t + 1 to avoid division by zero if t starts at 0
    m_hat = m / (1 - beta1**(t + 1))
    
    # 4. Compute bias-corrected second raw moment estimate
    v_hat = v / (1 - beta2**(t + 1))
    
    # 5. Update parameters
    param = param - lr * m_hat / (np.sqrt(v_hat) + eps)
    
    # Return the updated state so it can be used in the next iteration
    return param, m, v
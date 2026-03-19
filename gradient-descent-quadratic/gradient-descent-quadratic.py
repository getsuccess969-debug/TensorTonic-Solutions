def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    for epoch in range(steps):
        x0=x0-(2*a*x0+b)*lr
    return x0
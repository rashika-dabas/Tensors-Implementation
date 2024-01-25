def derive(f, x, h=0.001):
    return (f(x + h) - f(x)) / h # Numerical Derivation

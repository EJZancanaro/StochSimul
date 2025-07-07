import numpy as np
def supremum(f,a,b, precision=10.**(-5)) : #Trouve approx le sup d'une fonction f sur [a,b]
    """Returns an approximation of the supremum of a function on an interval
    params:
        f : function to evaluate
        a : float, lower bound of the interval
        b: float, upper bound of the interval
        precision: float, tolerated argmax error
    """


    X = np.arange(a,b, precision)
    return max(f(X))
import numpy as np

def softmax(x):
    """
    Compute the softmax of vector x.
    """
    # Subtract the maximum value of x from each element of x to avoid overflow
    # and make the softmax numerically stable.
    z = x - np.max(x)
    
    # Calculate the exponential of each element in z.
    exp_z = np.exp(z)
    
    # Calculate the sum of the exponential values.
    sum_exp_z = np.sum(exp_z)
    
    # Divide each element of the exponential vector by the sum of exponential values.
    softmax_z = exp_z / sum_exp_z
    
    return softmax_z
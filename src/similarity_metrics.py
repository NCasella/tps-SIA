import numpy as np 

"""
XP:input array de p elementos
W:matriz (n,p) de pesos (n: cantidad de neuronas)
"""

def _euclidean_distance(XP:np.ndarray,W:np.ndarray):
    distances= np.sqrt(np.sum((XP-W)**2,axis=1))
    return np.argmin(distances)

def _exponential_distance(XP:np.ndarray,W:np.ndarray):
    distances=np.exp(-np.sum((XP-W)**2,axis=1))
    return np.argmin(distances)

_metric_functions: dict[str,callable]={
    "euclidean_distance": _euclidean_distance,
    "exponential_distance":_exponential_distance
    }

def get_metric_function(string_function):
    return _metric_functions[string_function]



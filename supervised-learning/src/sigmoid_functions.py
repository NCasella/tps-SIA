import math
import numpy as np

def _tanh(beta,h):
    return np.tanh(beta*h)
def _logistic(beta,h):
    return 1/(1+math.exp(-2*beta*h))
def _relu(beta,h):
    return max(0,h)
def _softplus(beta,h):
    return math.log(1+math.exp(h))

def _tanh_derivate(beta,h):
    return beta*(1-_tanh(beta,h)**2.0)
def _logistic_derivate(beta,h):
    return 2*beta*_logistic(beta,h)*(1-_logistic(beta,h))
def _relu_derivate(beta,h):
    return 1 if h>0 else 0
def _softplus_derivate(beta,h):
    return math.exp(h)/(1+math.exp(h))

_sigmoid_functions:dict[str,tuple[callable,callable]]={
        "tanh":(_tanh,_tanh_derivate),
        "logistic":(_logistic,_logistic_derivate),
        "relu":(_relu,_relu_derivate),
        "softplus": (_softplus,_softplus_derivate)
}

def get_sigmoid_function_and_derivate(beta,function):
    sigmoid_function = lambda h: _sigmoid_functions[function][0](beta, h)
    sigmoid_derivate = lambda h: _sigmoid_functions[function][1](beta, h)
    return sigmoid_function, sigmoid_derivate

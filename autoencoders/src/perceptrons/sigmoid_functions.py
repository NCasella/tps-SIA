import numpy as np

def _tanh(beta, h):
    return np.tanh(beta * h)

def _logistic(beta, h):
    return 1 / (1 + np.exp(-2 * beta * h))

def _relu(beta, h):
    return np.maximum(0, h)

def _softplus(beta, h):
    return np.log(1 + np.exp(h))

def _tanh_derivate(beta, h):
    return beta * (1 - np.square(_tanh(beta, h)))

def _logistic_derivate(beta, h):
    sig = _logistic(beta, h)
    return 2 * beta * sig * (1 - sig)

def _relu_derivate(beta, h):
    return np.where(h > 0, 1.0, 0.0)

def _softplus_derivate(beta, h):
    return np.exp(h) / (1 + np.exp(h))

_sigmoid_functions: dict[str, tuple[callable, callable]] = {
    "tanh": (_tanh, _tanh_derivate),
    "logistic": (_logistic, _logistic_derivate),
    "relu": (_relu, _relu_derivate),
    "softplus": (_softplus, _softplus_derivate)
}

def get_sigmoid_function_and_derivate(beta, function):
    def sigmoid_function(h):
        return _sigmoid_functions[function][0](beta, h)

    def sigmoid_derivate(h):
        return _sigmoid_functions[function][1](beta, h)

    return sigmoid_function, sigmoid_derivate
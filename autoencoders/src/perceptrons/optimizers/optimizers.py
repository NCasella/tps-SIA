from src.perceptrons.optimizers.adam import Adam
from src.perceptrons.optimizers.gradient_descent import GradientDescent
from src.perceptrons.optimizers.momentum import Momentum

optimizers: dict = {
    "sgd": GradientDescent,
    "adam": Adam,
    "momentum": Momentum,
}

def get_optimizer(optimizer, learning_rate,alpha,beta1,beta2,epsilon, all_layers):
    return optimizers[optimizer](learning_rate, alpha,beta1,beta2,epsilon, all_layers)
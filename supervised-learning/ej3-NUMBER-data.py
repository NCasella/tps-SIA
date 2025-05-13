import random
import sys
import json

from src.optimizers.optimizers import get_optimizer
from src.sigmoid_functions import get_sigmoid_function_and_derivate
from src.multilayer_perceptron import MultilayerPerceptron
from src.confusion_matrix import ConfusionMatrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def read_number_files(file,block_size=35):
    with open(file, 'r') as f:
        text = f.read() 
    numbers = text.strip().split()  
    numbers = [int(num) for num in numbers]  
    blocks = [numbers[i:i+block_size] for i in range(0, len(numbers), block_size)]  
    blocks = [block for block in blocks if len(block) == block_size]    
    return blocks

def add_noise_to_dataset(dataset, noise_level):
    noisy_dataset = []
    for sample in dataset:
        noisy_sample = []
        for value in sample:
            if np.random.rand() < noise_level:
                noisy_sample.append(1 - value)  # Flip 0 to 1 and 1 to 0
            else:
                noisy_sample.append(value)
        noisy_dataset.append(noisy_sample)
    return noisy_dataset



if __name__ == "__main__":
    expected_outputs = np.identity(10)
    input_dataset = read_number_files("training/TP3-ej3-digitos.txt")
    layers = [40, 10]
    input_size = len(input_dataset[0])

    layer_shapes = []
    current_size = input_size + 1
    for layer_size in layers:
        layer_shapes.append((current_size, layer_size))
        current_size = layer_size + 1

    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    folder = "output"

    epochs = 1000
    epsilon = 0.02
    learning_rate = 0.001
    beta = 1
    optimizer_name = "adam"
    alpha = 0.001
    beta1 = 0.9
    beta2 = 0.999
    optimizer_epsilon = 0.00000008
    function_name = "tanh"

    function, derivate = get_sigmoid_function_and_derivate(beta, function_name)
    optimizer = get_optimizer(
        optimizer_name,
        learning_rate,
        alpha,
        beta1,
        beta2,
        optimizer_epsilon,
        layer_shapes
    )

    runs = 100
    steps = 100
    num_digits = 10

    noise_probabilities = [(1.0 / steps) * i for i in range(steps)]
    average_accuracy_per_digit = np.zeros((num_digits, steps))  # [digit][noise_level]

    for run in range(runs):
        perc = MultilayerPerceptron(
            learning_rate, input_dataset, expected_outputs,
            function, derivate, layers, optimizer
        )
        perc.train_perceptron(epochs=epochs, epsilon=epsilon)

        for idx, noise in enumerate(noise_probabilities):
            noisy_set = add_noise_to_dataset(input_dataset, noise)

            for digit_index in range(num_digits):
                confusion_matrix = ConfusionMatrix(labels)
                predicted = np.argmax(perc.predict_output(noisy_set[digit_index]))
                confusion_matrix.increment(labels[digit_index], predicted)
                average_accuracy_per_digit[digit_index][idx] += confusion_matrix.f1_score(digit_index)

    # Average over all runs
    average_accuracy_per_digit /= runs

    # Plot
    plt.figure()

    for digit_index in range(num_digits):
        plt.plot(
            noise_probabilities,
            average_accuracy_per_digit[digit_index],
            label=f'Digit {digit_index}'
        )

    plt.xlabel('Noise')
    plt.ylabel('F1 Score')
    plt.title(f'F1 score sobre cada digito con ruido (promedio sobre {runs} simulaciones)')
    plt.grid(True)
    plt.legend()
    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, 1.0)

    folder = "output"
    filename = "noise_average_per_digit.png"
    plt.savefig(f"{folder}/{filename}")
    plt.show()
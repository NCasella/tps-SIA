import random
import sys
import json

from src.optimizers.optimizers import get_optimizer
from src.sigmoid_functions import get_sigmoid_function_and_derivate
from src.multilayer_perceptron import MultilayerPerceptron
from src.confusion_matrix import ConfusionMatrix
import numpy as np
import pandas as pd

def read_number_files(file,block_size=35):
    with open(file, 'r') as f:
        text = f.read() 
    numbers = text.strip().split()  
    numbers = [int(num) for num in numbers]  
    blocks = [numbers[i:i+block_size] for i in range(0, len(numbers), block_size)]  
    blocks = [block for block in blocks if len(block) == block_size]    
    return blocks

if __name__=="__main__":
    
    xor_set=[[0, 0], [0, 1], [1, 0], [1, 1]]
    
    # por probabilidad 
    dataset_outputs={"XOR":[[0], [1], [1], [0]], "parity":[(1,0) if i%2==0 else (0,1) for i in range(10)] , "recognition":np.identity(10)} 
    
    with open(sys.argv[1],"r") as file:
         config=json.load(file)
    learning_rate=config["learning_rate"]
    epsilon=config["epsilon"]
    beta=config["beta"]
    problem=config["problem"]
    function=config["activation_function"]
    epochs=config["epochs"]
    optimizer_value=config["optimizer"]
    optimizer_alpha=config["optimizer_alpha"]
    optimizer_beta1=config["optimizer_beta_1"]
    optimizer_beta2=config["optimizer_beta_2"]
    optimizer_epsilon=config["optimizer_epsilon"]
    layers=config["layers"]
    training_file_path=config["training_file_path"]
    test_indices = config["testing_indices"]
    test_noise = config["test_noise"]
    if problem=="XOR":
        input_dataset=xor_set
    else:
        input_dataset=read_number_files(training_file_path)
    
    activation_function,activation_derivate=get_sigmoid_function_and_derivate(beta,function)

    expected_outputs=dataset_outputs[config["problem"]]

    testing_input = [input_dataset[i] for i in test_indices]
    testing_output = [expected_outputs[i] for i in test_indices]

    training_input = [input_dataset[i] for i in range(len(input_dataset)) if i not in test_indices]
    training_output = [expected_outputs[i] for i in range(len(expected_outputs)) if i not in test_indices]

    print(f"Input dataset: {input_dataset}")
    print(f"Expected outputs: {expected_outputs}")
    print(f"Layers: {layers}")

    input_size = len(input_dataset[0])
    layer_shapes = []
    current_size = input_size + 1
    for layer_size in layers:
        layer_shapes.append((current_size, layer_size))
        current_size = layer_size + 1

    optimizer = get_optimizer(
        optimizer_value,
        learning_rate,
        optimizer_alpha,
        optimizer_beta1,
        optimizer_beta2,
        optimizer_epsilon,
        layer_shapes
    )

    perceptron:MultilayerPerceptron =MultilayerPerceptron(learning_rate,training_input,training_output,activation_function,activation_derivate,layers, optimizer)
    perceptron.train_perceptron(epochs=epochs,epsilon=epsilon)

    print("Training set")
    for i,input in enumerate(training_input):
        print(f"Input: {input}, predicted: {perceptron.predict_output(input)}, expected: {training_output[i]}")

    print("Testing set:")
    for j,input in enumerate(testing_input):
        print(f"Input: {input}, predicted: {perceptron.predict_output(input)}, expected: {training_output[i]}")

    noisy_set = []
    for input in training_input:
        noisy_input = []
        for value in input:
            if random.random() < test_noise:
                noisy_input.append(1 - value)
            else:
                noisy_input.append(value)
        noisy_set.append(noisy_input)

    expected_values = np.argmax(expected_outputs, axis=1)
    noisy_confusion_matrix = ConfusionMatrix(expected_values)

    print(f"Noisy set ({test_noise}):")
    for k, input in enumerate(noisy_set):
        noisy_output = perceptron.predict_output(input)
        max_index = np.argmax(noisy_output)
        noisy_confusion_matrix.increment(expected_values[k], max_index)
        print(f"Input: {input}, predicted: {max_index}, expected: {np.argmax(training_output[k])}")

    print(f"Accuracy: {noisy_confusion_matrix.accuracy()}")


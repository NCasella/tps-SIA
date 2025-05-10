import random
import sys
import json
from src.sigmoid_functions import get_sigmoid_function_and_derivate
from src.multilayer_perceptron import MultilayerPerceptron
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
    dataset_outputs={"XOR":[[0], [1], [1], [0]], "parity":[i%2 for i in range(9)] , "recognition":[i for i in range(9)]} 
    
    with open(sys.argv[1],"r") as file:
         config=json.load(file)
    learning_rate=config["learning_rate"]
    epsilon=config["epsilon"]
    beta=config["beta"]
    problem=config["problem"]
    function=config["activation_function"]
    epochs=config["epochs"]
    layers=config["layers"]
    training_file_path=config["training_file_path"]
    if problem=="XOR":
        input_dataset=xor_set
    else:
        input_dataset=read_number_files(training_file_path)
    
    activation_function,activation_derivate=get_sigmoid_function_and_derivate(beta,function)
    
    expected_outputs=dataset_outputs[config["problem"]]

    print(f"Input dataset: {input_dataset}")
    print(f"Expected outputs: {expected_outputs}")
    print(f"Layers: {layers}")
    
    perceptron:MultilayerPerceptron =MultilayerPerceptron(learning_rate,input_dataset,expected_outputs,activation_function,activation_derivate,layers)
    perceptron.train_perceptron(epochs=epochs,epsilon=epsilon)
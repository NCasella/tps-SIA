import numpy as np 
import pandas as pd 
import sys
import json 
from src.hopfield import Hopfield
import matplotlib.pyplot as plt

def read_number_files(file,block_size=25):
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
                noisy_sample.append(-1 if value==1 else 1) 
            else:
                noisy_sample.append(value)
        noisy_dataset.append(noisy_sample)
    return noisy_dataset

if __name__=="__main__":
    with open(sys.argv[1],"r") as f:
        config=json.load(f)
    
    
    letters=read_number_files(config["data_source"])
    iterations=config["iterations"]
    learning_rate=config["learning_rate"]
    constant_learning_rate=config["constant_learning_rate"]
    noise_rate=config["noise"]

    hopfield: Hopfield= Hopfield(letters)
    letter_states=[]
    energy_per_input=[]
    
    
    noisy_letters=add_noise_to_dataset(letters,noise_rate)
    for letter in noisy_letters:
        states,energy_history=hopfield.predic_output(letter,iterations)
        print(f"{states[-1].reshape(5,5)} energy:{energy_history}")
        
    
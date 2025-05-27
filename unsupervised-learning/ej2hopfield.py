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

if __name__=="__main__":
    with open(sys.argv[1],"r") as f:
        config=json.load(f)
    
    
    letters=read_number_files(config["data_source"])
    iterations=config["iterations"]
    learning_rate=config["learning_rate"]
    constant_learning_rate=config["constant_learning_rate"]

    hopfield: Hopfield= Hopfield(letters)
    letter_states=[]
    energy_per_input=[]
    
    for letter in letters:
        states,energy_history=hopfield.predic_output(letter,iterations)#TODO: agregarle ruido
        print(f"{states[-1].reshape(5,5)} energy:{energy_history}")
        
    
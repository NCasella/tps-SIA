import sys
import json
import math
from src.linear_perceptron import LinearPerceptron
from src.non_linear_perceptron import NonLinearPerceptron
from src.sigmoid_functions import get_sigmoid_function_and_derivate
import numpy as np 
import pandas as pd 

if __name__=="__main__":
    with open(sys.argv[1],"r") as file:
        config=json.load(file)
    
    learning_rate:float=config["learning_rate"]
    epsilon:int=config["epsilon"]
    function:str=config["activation_function"]
    epochs:int=config["epochs"]
    perceptron_type:str=config["perceptron_type"]
    training_file_path:str=config["training_file_path"]
    beta:float=config["beta"]

    df=pd.read_csv(training_file_path)

    sigmoid_function,sigmoid_derivate= get_sigmoid_function_and_derivate(beta,function)

    perceptrons_map:dict[str:callable]={
        "linear": lambda rate, training_input, training_output:LinearPerceptron(learning_rate=rate, training_input=training_input, training_output=training_output),
        "non_linear":lambda rate, training_input, training_output:NonLinearPerceptron(learning_rate=rate, training_input=training_input, training_output=training_output, activation_function=sigmoid_function, activation_function_derivate=sigmoid_derivate)
    }
    
    perceptron: LinearPerceptron|NonLinearPerceptron=perceptrons_map[perceptron_type](learning_rate,df[["x1","x2","x3"]],df["y"])
    perceptron.train_perceptron(epochs=epochs,epsilon=epsilon)
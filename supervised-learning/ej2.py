import sys
import json
import math
from src.linear_perceptron import LinearPerceptron
from src.non_linear_perceptron import NonLinearPerceptron
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

    sigmoid_functions:dict[str,callable]={
        "tanh":lambda h: math.tanh(beta*h),
        "logistic":lambda h:1/(1+math.exp(-2*beta*h))
        }

    sigmoid_function:callable=sigmoid_functions[function]
    
    sigmoid_derivates:dict[str,callable]={
        "tanh":lambda h:beta*(1-sigmoid_function(h)),
        "logistic":lambda h:2*beta*sigmoid_function(h)*(1-sigmoid_function(h))
    }
    sigmoid_derivate=sigmoid_derivates[function]


    perceptrons_map:dict[str:callable]={
        "linear": lambda rate, training_input, training_output:LinearPerceptron(learning_rate=rate, training_input=training_input, training_output=training_output),
        "non_linear":lambda rate, training_input, training_output:NonLinearPerceptron(learning_rate=rate, training_input=training_input, training_output=training_output, activation_function=sigmoid_function, activation_function_derivate=sigmoid_derivate)
        }
    
    perceptron: LinearPerceptron|NonLinearPerceptron=perceptrons_map[perceptron_type](learning_rate,df[["x1","x2","x3"]],df["y"])

    perceptron.train_perceptron(epochs=epochs,epsilon=epsilon)
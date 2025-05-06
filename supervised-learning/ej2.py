import sys
import json
import math
from src.linear_perceptron import LinearPerceptron
from src.non_linear_perceptron import NonLinearPerceptron
from src.sigmoid_functions import get_sigmoid_function_and_derivate
import numpy as np 
import pandas as pd 
import random
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
    testing_set_size=config["testing_set_size"]

    df=pd.read_csv(training_file_path)
    max_y=df["y"].max()
    min_y=df["y"].min()
    for i,y in enumerate(df["y"]):
        df["y"][i]=(y-min_y)/(max_y-min_y)*2-1
        
    indexes=list(range(len(df)))
    random.shuffle(indexes)
    testing_indexes = indexes[:testing_set_size]
    training_indexes = indexes[testing_set_size:]
    
    df_array = np.array(df)
    
    training_rows_x = []
    training_rows_y = []
    for x in training_indexes:
        training_rows_x.append(df_array[x][:-1])
        training_rows_y.append(df_array[x][-1])
    
    
    testing_rows_x = []
    testing_rows_y = []
    for x in testing_indexes:
        testing_rows_x.append(df_array[x][:-1])
        testing_rows_y.append(df_array[x][-1])

    testing_rows = [df_array[x] for x in testing_indexes]
    
    sigmoid_function,sigmoid_derivate= get_sigmoid_function_and_derivate(beta,function)

    perceptrons_map:dict[str:callable]={
        "linear": lambda rate, training_input, training_output:LinearPerceptron(learning_rate=rate, training_input=training_input, training_output=training_output),
        "non_linear":lambda rate, training_input, training_output:NonLinearPerceptron(learning_rate=rate, training_input=training_input, training_output=training_output, activation_function=sigmoid_function, activation_function_derivate=sigmoid_derivate)
    }
    print(df)
    print(len(df))
#
    perceptron: LinearPerceptron|NonLinearPerceptron=perceptrons_map[perceptron_type](learning_rate,training_rows_x,training_rows_y)
    perceptron.train_perceptron(epochs=epochs,epsilon=epsilon)
    
    for i,x in enumerate(testing_rows_x):
        output=perceptron.predict_output(x)
        print(f"{output} vs {testing_rows_y[i]} error: {perceptron.calculate_error(testing_rows_y[i],output)}")

import sys
import json
from src.simple_perceptron import SimplePerceptron

dataset = ((-1,-1), (-1, 1), (1,-1), (1,1))
and_output = (-1, -1, -1, 1)
xor_output = (-1, 1, 1, -1)
training_output_map={
    "AND":and_output,
    "XOR":xor_output
    }    

if __name__=="__main__":
    with open(sys.argv[1],"r") as file:
        config=json.load(file)
    learning_rate=config["learning_rate"]
    epsilon=config["epsilon"]
    function=config["function"]
    epochs=config["epochs"]
    perc=SimplePerceptron(learning_rate=learning_rate,training_input=dataset,training_output=training_output_map[function])
    perc.train_perceptron(epochs=epochs,epsilon=epsilon)
    print(perc.predict_output((-1, -1)))
    print(perc.predict_output((-1, 1)))
    print(perc.predict_output((1, -1)))
    print(perc.predict_output((1, 1)))
    

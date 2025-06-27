import sys
import json
from src.simple_perceptron import SimplePerceptron
from src.confusion_matrix import ConfusionMatrix
import matplotlib.pyplot as plt

def run_perceptron(epsilon, learning_rate, epochs, training_input, training_output):
    perc = SimplePerceptron(learning_rate=learning_rate, training_input=training_input,
                            training_output=training_output)
    perc.train_perceptron(epochs=epochs, epsilon=epsilon)

    return perc

dataset = ((-1,-1), (-1, 1), (1,-1), (1,1))
and_output = (-1, -1, -1, 1)

if __name__=="__main__":

    learning_rate_accuracy = []

    y = [0, 0]

    for i in range(1000):
        output = run_perceptron(0.02, 0.005, 100, dataset[1:], and_output[1:]).predict_output((-1, -1))
        if (output == -1):
            y[0] += 1
        if (output == 1):
            y[1] += 1



    x = ["0", "1"]

    plt.bar(x, y)
    plt.show()





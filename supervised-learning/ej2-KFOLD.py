import sys
import json
import math
from src.linear_perceptron import LinearPerceptron
from src.non_linear_perceptron import NonLinearPerceptron
from src.sigmoid_functions import get_sigmoid_function_and_derivate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

def train_with_k_fold_leave_one_out(k_folds_input, k_folds_output, learning_rate, epochs, epsilon):

    perceptron = NonLinearPerceptron(
        learning_rate=learning_rate,
        training_input=k_folds_input,
        training_output=k_folds_output,
        activation_function=sigmoid_function,
        activation_function_derivate=sigmoid_derivate
    )

    perceptron.train_perceptron(epochs=epochs, epsilon=epsilon)

    return perceptron

if __name__ == "__main__":

    learning_rate = 0.00005
    epsilon = 0.02
    function = "tanh"
    epochs = 1000
    training_file_path = "training/TP3-ej2-conjunto.csv"
    beta = 1
    sigmoid_function, sigmoid_derivate = get_sigmoid_function_and_derivate(beta, function)

    df = pd.read_csv(training_file_path)
    max_y = df["y"].max()
    min_y = df["y"].min()
    for i, y in enumerate(df["y"]):
        df["y"][i] = (y - min_y) / (max_y - min_y)

    k = 28
    k_folds = []

    base_size = len(df) // k
    extra = len(df) % k

    start = 0
    num_runs = 100
    fold_errors = [0.0 for _ in range(k)]

    for run in range(num_runs):

        df_shuffled = df.sample(frac=1).reset_index(drop=True)


        k_folds = []
        base_size = len(df_shuffled) // k
        extra = len(df_shuffled) % k

        start = 0
        for i in range(k):
            fold_size = base_size + (1 if i < extra else 0)
            k_folds.append(df_shuffled.iloc[start:start + fold_size])
            start += fold_size


        for fold_to_leave_out in range(k):
            training_input = []
            training_output = []
            testing_input = []
            testing_output = []

            for i, fold in enumerate(k_folds):
                fold_np = fold.to_numpy()
                if i != fold_to_leave_out:
                    training_input.extend(fold_np[:, :-1])
                    training_output.extend(fold_np[:, -1])
                else:
                    testing_input.extend(fold_np[:, :-1])
                    testing_output.extend(fold_np[:, -1])

            training_input = np.array(training_input)
            training_output = np.array(training_output)
            testing_input = np.array(testing_input)
            testing_output = np.array(testing_output)
            perc = train_with_k_fold_leave_one_out(training_input, training_output, learning_rate, epochs, epsilon)

            total_error = 0
            for i, input in enumerate(testing_input):
                predicted = perc.predict_output(input)
                total_error += (testing_output[i] - predicted)

            fold_errors[fold_to_leave_out] += abs(total_error / len(testing_input))
        print(f"{run}/{num_runs}")

    avg_errors = []

    for i, err in enumerate(fold_errors):
        avg_error = err / num_runs
        print(f"Fold {i + 1} average error over {num_runs} runs: {avg_error}")
        avg_errors.append(avg_error)

    labels = []
    for i in range(k):
        labels.append(i+1)

    plt.xlabel('K')
    plt.ylabel('Error Promedio')
    plt.title('Error promedio por K conjunto')

    plt.legend()
    plt.ylim(0.0, 1.0)
    plt.bar(labels, avg_errors)
    plt.show()






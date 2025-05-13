import sys
import json
import numpy as np 
import pandas as pd 
import random
import matplotlib.pyplot as plt
from src.linear_perceptron import LinearPerceptron
from src.non_linear_perceptron import NonLinearPerceptron
from src.sigmoid_functions import get_sigmoid_function_and_derivate

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
    df_tanh = df.copy()
    df_logistic = df.copy()
    max_y=df["y"].max()
    min_y=df["y"].min()
    # normalizar entre -1 y 1
    for i,y in enumerate(df_tanh["y"]):
        df_tanh["y"][i]=(y-min_y)/(max_y-min_y)*2-1
    # normalizar entre 0 y 1
    for i,y in enumerate(df_logistic["y"]):
        df_logistic["y"][i]=(y-min_y)/(max_y-min_y)

    def _reverse_tanh_normalize(value):
        return (value + 1) * (max_y - min_y) / 2 + min_y
        
    indexes=list(range(len(df)))
    random.shuffle(indexes)
    testing_indexes = indexes[:testing_set_size]
    training_indexes = indexes[testing_set_size:]
    
    df_array = np.array(df)
    df_array_tanh = np.array(df_tanh)
    df_array_logistic = np.array(df_logistic)
    
    # sin normalizar
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

    # normalizado entre -1 y 1
    training_rows_x_tanh = []
    training_rows_y_tanh = []
    for x in training_indexes:
        training_rows_x_tanh.append(df_array_tanh[x][:-1])
        training_rows_y_tanh.append(df_array_tanh[x][-1])
    
    testing_rows_x_tanh = []
    testing_rows_y_tanh = []
    for x in testing_indexes:
        testing_rows_x_tanh.append(df_array_tanh[x][:-1])
        testing_rows_y_tanh.append(df_array_tanh[x][-1])
    
    testing_rows_tanh = [df_array_tanh[x] for x in testing_indexes]

    # normalizado entre 0 y 1
    training_rows_x_logistic = []
    training_rows_y_logistic = []
    for x in training_indexes:
        training_rows_x_logistic.append(df_array_logistic[x][:-1])
        training_rows_y_logistic.append(df_array_logistic[x][-1])

    testing_rows_x_logistic = []
    testing_rows_y_logistic = []
    for x in testing_indexes:
        testing_rows_x_logistic.append(df_array_logistic[x][:-1])
        testing_rows_y_logistic.append(df_array_logistic[x][-1])

    testing_rows_logistic = [df_array_logistic[x] for x in testing_indexes]
    
    # sigmoid_function,sigmoid_derivate= get_sigmoid_function_and_derivate(beta,function)

    tanh, dtanh = get_sigmoid_function_and_derivate(beta, "tanh")
    logistic, dlogistic = get_sigmoid_function_and_derivate(beta, "logistic")
    relu, drelu = get_sigmoid_function_and_derivate(beta, "relu")
    softplus, dsoftplus = get_sigmoid_function_and_derivate(beta, "softplus")

    # print(df)
    # print(len(df))

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def _get_input_with_bias(training_input):
        training_input = np.array(training_input)
        if training_input.ndim == 1:  
            training_input = training_input.reshape(1, -1)
        bias = np.ones((training_input.shape[0], 1), dtype=int)  
        return np.hstack([bias, training_input])  

    training_input=_get_input_with_bias(training_rows_x_tanh)
    weights = [ random.uniform(-0.5,0.5) for _ in range(training_input.shape[1]) ]

    tanh_perceptron001: NonLinearPerceptron = NonLinearPerceptron(1e-2,training_rows_x_tanh,training_rows_y_tanh,activation_function=tanh,activation_function_derivate=dtanh)    
    tanh_perceptron0001: NonLinearPerceptron = NonLinearPerceptron(1e-2,training_rows_x_tanh,training_rows_y_tanh,activation_function=tanh,activation_function_derivate=dtanh)
    tanh_perceptron00001: NonLinearPerceptron = NonLinearPerceptron(1e-2,training_rows_x_tanh,training_rows_y_tanh,activation_function=tanh,activation_function_derivate=dtanh)

    # tanh_perceptron001.weights = weights.copy()
    # tanh_perceptron0001.weights = weights.copy()
    # tanh_perceptron00001.weights = weights.copy()

    testing_array001 = []
    testing_array0001 = []
    testing_array00001 = []

    training_array001 = []
    training_array0001 = []
    training_array00001 = []

    for i in range(int(epochs/5)):
        tanh_perceptron001.train_perceptron(epochs=5,epsilon=epsilon)    
        tanh_perceptron0001.train_perceptron(epochs=5,epsilon=epsilon)
        tanh_perceptron00001.train_perceptron(epochs=5,epsilon=epsilon)

        errors001 = []
        errors0001 = []
        errors00001 = []
        for j, x in enumerate(testing_rows_x_tanh):
            expected = testing_rows_y_tanh[j]

            # error_rate = abs((expected - tanh_perceptron001.predict_output(x)) / expected)
            # errors001.append(error_rate)

            # error_rate = abs((expected - tanh_perceptron0001.predict_output(x)) / expected)
            # errors0001.append(error_rate)

            # error_rate = abs((expected - tanh_perceptron00001.predict_output(x)) / expected)
            # errors00001.append(error_rate)

            error = tanh_perceptron001.calculate_error(expected, tanh_perceptron001.predict_output(x))
            errors001.append(error)

            error = tanh_perceptron0001.calculate_error(expected, tanh_perceptron0001.predict_output(x))
            errors0001.append(error)

            error = tanh_perceptron00001.calculate_error(expected, tanh_perceptron00001.predict_output(x))
            errors00001.append(error)

        testing_array001.append(np.mean(errors001))
        testing_array0001.append(np.mean(errors0001))
        testing_array00001.append(np.mean(errors00001))

        errors001 = []
        errors0001 = []
        errors00001 = []
        for j, x in enumerate(training_rows_x_tanh):
            expected = training_rows_y_tanh[j]

            # error_rate = abs((expected - tanh_perceptron001.predict_output(x)) / expected)
            # errors001.append(error_rate)

            # error_rate = abs((expected - tanh_perceptron0001.predict_output(x)) / expected)
            # errors0001.append(error_rate)

            # error_rate = abs((expected - tanh_perceptron00001.predict_output(x)) / expected)
            # errors00001.append(error_rate)

            error = tanh_perceptron001.calculate_error(expected, tanh_perceptron001.predict_output(x))
            errors001.append(error)

            error = tanh_perceptron0001.calculate_error(expected, tanh_perceptron0001.predict_output(x))
            errors0001.append(error)

            error = tanh_perceptron00001.calculate_error(expected, tanh_perceptron00001.predict_output(x))
            errors00001.append(error)

        training_array001.append(np.mean(errors0001))
        training_array0001.append(np.mean(errors0001))
        training_array00001.append(np.mean(errors00001))

    x = [i*5 for i in range(int(epochs/5))]
    average_test = np.mean([testing_array001, testing_array0001, testing_array00001], axis=0)
    average_train = np.mean([training_array001, training_array0001, training_array00001], axis=0)
    plt.plot(x, average_test, label="test")
    plt.plot(x, average_train, label="train")

    # plt.plot(x, testing_array001, label="1e-2 test", color="red")
    # plt.plot(x, training_array001, label="1e-2 train", color="orange")
    # plt.plot(x, testing_array0001, label="1e-3 test", color="blue")
    # plt.plot(x, training_array0001, label="1e-3 train", color="cyan")
    # plt.plot(x, testing_array00001, label="1e-4 test", color="green")
    # plt.plot(x, training_array00001, label="1e-4 train", color="lightgreen")
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.title("Error for learning rate 1e-2")
    plt.legend()
    plt.ylim(0, 0.1)
    # plt.yscale("log")
    plt.show()


    # print(f"Error after {epochs} epochs for tanh perceptron with learning rate 1: {tanh_training_error_history1[-1]}")
    # print(f"Error after {epochs} epochs for tanh perceptron with learning rate 0.1: {tanh_training_error_history01[-1]}")

    # print(f"Error after {epochs} epochs for tanh perceptron with learning rate 1e-2: {tanh_training_error_history001[-1]}")
    # print(f"Error after {epochs} epochs for tanh perceptron with learning rate 1e-3: {tanh_training_error_history0001[-1]}")
    # print(f"Error after {epochs} epochs for tanh perceptron with learning rate 1e-4: {tanh_training_error_history00001[-1]}")

    # for i,x in enumerate(testing_rows_x):
    #     output=tanh_perceptron001.predict_output(x)
    #     print(f"Learning rate 1e-2:\noutput:{_reverse_tanh_normalize(output)} - expected:{_reverse_tanh_normalize(testing_rows_y_tanh[i])} - error:{tanh_perceptron001.calculate_error(testing_rows_y_tanh[i],output)}")

    #     output=tanh_perceptron0001.predict_output(x)
    #     print(f"Learning rate 1e-3:\noutput:{_reverse_tanh_normalize(output)} - expected:{_reverse_tanh_normalize(testing_rows_y_tanh[i])} - error:{tanh_perceptron0001.calculate_error(testing_rows_y_tanh[i],output)}")

    #     output=tanh_perceptron00001.predict_output(x)
    #     print(f"Learning rate 1e-4:\noutput:{_reverse_tanh_normalize(output)} - expected:{_reverse_tanh_normalize(testing_rows_y_tanh[i])} - error:{tanh_perceptron00001.calculate_error(testing_rows_y_tanh[i],output)}")


    # print(f"Average error after {epochs} epochs for learning rate 1: {np.mean(array1)}")
    # print(f"Average error after {epochs} epochs for learning rate 0.1: {np.mean(array01)}")
    # print(f"Average error after {epochs} epochs for learning rate 1e-2: {np.mean(array001)}")
    # print(f"Average error after {epochs} epochs for learning rate 1e-3: {np.mean(array0001)}")
    # print(f"Average error after {epochs} epochs for learning rate 1e-4: {np.mean(array00001)}")

    # logistic_perceptron1: NonLinearPerceptron = NonLinearPerceptron(1,training_rows_x_logistic,training_rows_y_logistic,activation_function=logistic,activation_function_derivate=dlogistic)
    # logistic_perceptron01: NonLinearPerceptron = NonLinearPerceptron(0.1,training_rows_x_logistic,training_rows_y_logistic,activation_function=logistic,activation_function_derivate=dlogistic)
    # logistic_perceptron001: NonLinearPerceptron = NonLinearPerceptron(1e-2,training_rows_x_logistic,training_rows_y_logistic,activation_function=logistic,activation_function_derivate=dlogistic)
    # logistic_perceptron0001: NonLinearPerceptron = NonLinearPerceptron(1e-3,training_rows_x_logistic,training_rows_y_logistic,activation_function=logistic,activation_function_derivate=dlogistic)
    # logistic_perceptron00001: NonLinearPerceptron = NonLinearPerceptron(1e-4,training_rows_x_logistic,training_rows_y_logistic,activation_function=logistic,activation_function_derivate=dlogistic)

    # logistic_perceptron1.weights = weights.copy()
    # logistic_perceptron01.weights = weights.copy()
    # logistic_perceptron001.weights = weights.copy()
    # logistic_perceptron0001.weights = weights.copy()
    # logistic_perceptron00001.weights = weights.copy()

    # logistic_training_error_history1, logistic_convergence1 = logistic_perceptron1.train_perceptron(epochs=epochs,epsilon=epsilon)
    # logistic_training_error_history01, logistic_convergence01 = logistic_perceptron01.train_perceptron(epochs=epochs,epsilon=epsilon)
    # logistic_training_error_history001, logistic_convergence001 = logistic_perceptron001.train_perceptron(epochs=epochs,epsilon=epsilon)
    # logistic_training_error_history0001, logistic_convergence0001 = logistic_perceptron0001.train_perceptron(epochs=epochs,epsilon=epsilon)
    # logistic_training_error_history00001, logistic_convergence00001 = logistic_perceptron00001.train_perceptron(epochs=epochs,epsilon=epsilon)

    # print(f"Error after {epochs} epochs for logistic perceptron with learning rate 1: {logistic_training_error_history1[-1]}")
    # print(f"Error after {epochs} epochs for logistic perceptron with learning rate 0.1: {logistic_training_error_history01[-1]}")
    # print(f"Error after {epochs} epochs for logistic perceptron with learning rate 1e-2: {logistic_training_error_history001[-1]}")
    # print(f"Error after {epochs} epochs for logistic perceptron with learning rate 1e-3: {logistic_training_error_history0001[-1]}")
    # print(f"Error after {epochs} epochs for logistic perceptron with learning rate 1e-4: {logistic_training_error_history00001[-1]}")

    # plt.plot(logistic_training_error_history1, label=r'$\eta = 1$')
    # plt.plot(logistic_training_error_history01, label=r'$\eta = 0.1$')
    # plt.plot(logistic_training_error_history001, label=r'$\eta = 1e-2$')
    # plt.plot(logistic_training_error_history0001, label=r'$\eta = 1e-3$')
    # plt.plot(logistic_training_error_history00001, label=r'$\eta = 1e-4$')

    # plt.title("Logistic function with varying learning rates")

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # plt.ylim(0, 10)
    # plt.xlabel('Epochs')
    # plt.ylabel('Error')
    # plt.legend(loc='upper right')
    # plt.show()

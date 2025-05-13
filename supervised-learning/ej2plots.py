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

    # different functions
    # tanh_perceptron: NonLinearPerceptron = NonLinearPerceptron(learning_rate,training_rows_x_tanh,training_rows_y_tanh,activation_function=tanh,activation_function_derivate=dtanh)
    # logistic_perceptron: NonLinearPerceptron = NonLinearPerceptron(learning_rate,training_rows_x_logistic,training_rows_y_logistic,activation_function=logistic,activation_function_derivate=dlogistic)
    # relu_perceptron: NonLinearPerceptron = NonLinearPerceptron(learning_rate,training_rows_x,training_rows_y,activation_function=relu,activation_function_derivate=drelu)
    # softplus_perceptron: NonLinearPerceptron = NonLinearPerceptron(learning_rate,training_rows_x,training_rows_y,activation_function=softplus,activation_function_derivate=dsoftplus)

    # for average
    # non_linear_perceptron: NonLinearPerceptron = NonLinearPerceptron(learning_rate,training_rows_x,training_rows_y,activation_function=sigmoid_function,activation_function_derivate=sigmoid_derivate)
    # non_linear_perceptron2: NonLinearPerceptron = NonLinearPerceptron(learning_rate,training_rows_x,training_rows_y,activation_function=sigmoid_function,activation_function_derivate=sigmoid_derivate)
    # non_linear_perceptron3: NonLinearPerceptron = NonLinearPerceptron(learning_rate,training_rows_x,training_rows_y,activation_function=sigmoid_function,activation_function_derivate=sigmoid_derivate)
    # non_linear_perceptron4: NonLinearPerceptron = NonLinearPerceptron(learning_rate,training_rows_x,training_rows_y,activation_function=sigmoid_function,activation_function_derivate=sigmoid_derivate)

    # linear_perceptron: LinearPerceptron = LinearPerceptron(learning_rate,training_rows_x,training_rows_y)
    # linear_perceptron2: LinearPerceptron = LinearPerceptron(learning_rate,training_rows_x,training_rows_y)
    # linear_perceptron3: LinearPerceptron = LinearPerceptron(learning_rate,training_rows_x,training_rows_y)
    # linear_perceptron4: LinearPerceptron = LinearPerceptron(learning_rate,training_rows_x,training_rows_y)

    # non_linear_training_error_history = non_linear_perceptron.train_perceptron(epochs=epochs,epsilon=epsilon)

    # ALL FUNCTIONS

    # tanh_training_error_history, tanh_convergence = tanh_perceptron.train_perceptron(epochs=epochs,epsilon=epsilon)
    # logistic_training_error_history, logistic_convergence = logistic_perceptron.train_perceptron(epochs=epochs,epsilon=epsilon)
    # relu_training_error_history, relu_convergence = relu_perceptron.train_perceptron(epochs=epochs,epsilon=epsilon)
    # softplus_training_error_history, softplus_convergence = softplus_perceptron.train_perceptron(epochs=epochs,epsilon=epsilon)
    # linear_training_error_history, linear_convergence = linear_perceptron.train_perceptron(epochs=epochs,epsilon=epsilon)

    # tanh_average = np.mean(tanh_training_error_history)
    # tanh_standard_deviation = np.std(tanh_training_error_history)
    # logistic_average = np.mean(logistic_training_error_history)
    # logistic_standard_deviation = np.std(logistic_training_error_history)
    # relu_average = np.mean(relu_training_error_history)
    # relu_standard_deviation = np.std(relu_training_error_history)
    # softplus_average = np.mean(softplus_training_error_history)
    # softplus_standard_deviation = np.std(softplus_training_error_history)
    # linear_average = np.mean(linear_training_error_history)
    # linear_standard_deviation = np.std(linear_training_error_history)

    # x = ["tanh", "logistic", "relu", "softplus", "linear"]
    # y = [tanh_average, logistic_average, relu_average, softplus_average, linear_average]
    # e = [tanh_standard_deviation, logistic_standard_deviation, relu_standard_deviation, softplus_standard_deviation, linear_standard_deviation]

    # print(f"tanh convergence: {tanh_convergence}\nlogistic convergence: {logistic_convergence}\nrelu convergence: {relu_convergence}\nsoftplus convergence: {softplus_convergence}\nlinear convergence: {linear_convergence}")

    # plt.errorbar(x, y, yerr=e, fmt='o', capsize=5)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # array1 = []
    # array01 = []
    # array001 = []
    # array0001 = []
    # array00001 = []

    # for i in range(5):
    #     # different learning rates

    def _get_input_with_bias(training_input):
        training_input = np.array(training_input)
        if training_input.ndim == 1:  
            training_input = training_input.reshape(1, -1)
        bias = np.ones((training_input.shape[0], 1), dtype=int)  
        return np.hstack([bias, training_input])  

    training_input=_get_input_with_bias(training_rows_x_logistic)
    weights = [ random.uniform(-0.5,0.5) for w in range(training_input.shape[1]) ]

    # # tanh_perceptron1: NonLinearPerceptron = NonLinearPerceptron(1,training_rows_x_tanh,training_rows_y_tanh,activation_function=tanh,activation_function_derivate=dtanh)
    # # tanh_perceptron01: NonLinearPerceptron = NonLinearPerceptron(0.1,training_rows_x_tanh,training_rows_y_tanh,activation_function=tanh,activation_function_derivate=dtanh)
    # tanh_perceptron001: NonLinearPerceptron = NonLinearPerceptron(1e-2,training_rows_x_tanh,training_rows_y_tanh,activation_function=tanh,activation_function_derivate=dtanh)    
    # tanh_perceptron0001: NonLinearPerceptron = NonLinearPerceptron(1e-3,training_rows_x_tanh,training_rows_y_tanh,activation_function=tanh,activation_function_derivate=dtanh)
    # tanh_perceptron00001: NonLinearPerceptron = NonLinearPerceptron(1e-4,training_rows_x_tanh,training_rows_y_tanh,activation_function=tanh,activation_function_derivate=dtanh)

    # # tanh_perceptron1.weights = weights.copy()
    # # tanh_perceptron01.weights = weights.copy()
    # tanh_perceptron001.weights = weights.copy()
    # tanh_perceptron0001.weights = weights.copy()
    # tanh_perceptron00001.weights = weights.copy()

    # # tanh_training_error_history1, tanh_convergence1 = tanh_perceptron1.train_perceptron(epochs=epochs,epsilon=epsilon)
    # # tanh_training_error_history01, tanh_convergence2 = tanh_perceptron01.train_perceptron(epochs=epochs,epsilon=epsilon)
    
    # tanh_training_error_history001, tanh_convergence = tanh_perceptron001.train_perceptron(epochs=epochs,epsilon=epsilon)    
    # tanh_training_error_history0001, tanh_convergence3 = tanh_perceptron0001.train_perceptron(epochs=epochs,epsilon=epsilon)
    # tanh_training_error_history00001, tanh_convergence4 = tanh_perceptron00001.train_perceptron(epochs=epochs,epsilon=epsilon)

    # # print(f"Error after {epochs} epochs for tanh perceptron with learning rate 1: {tanh_training_error_history1[-1]}")
    # # print(f"Error after {epochs} epochs for tanh perceptron with learning rate 0.1: {tanh_training_error_history01[-1]}")
    # print(f"Error after {epochs} epochs for tanh perceptron with learning rate 1e-2: {tanh_training_error_history001[-1]}")
    # print(f"Error after {epochs} epochs for tanh perceptron with learning rate 1e-3: {tanh_training_error_history0001[-1]}")
    # print(f"Error after {epochs} epochs for tanh perceptron with learning rate 1e-4: {tanh_training_error_history00001[-1]}")

    # # plt.plot(tanh_training_error_history1, label=r'$\eta = 1$')
    # # plt.plot(tanh_training_error_history01, label=r'$\eta = 0.1$')
    # plt.plot(tanh_training_error_history001, label=r'$\eta = 1e-2$')
    # plt.plot(tanh_training_error_history0001, label=r'$\eta = 1e-3$')
    # plt.plot(tanh_training_error_history00001, label=r'$\eta = 1e-4$')

    # plt.title("Tanh function with varying learning rates")
    # plt.xlabel('Epochs')
    # plt.ylabel('Error')
    # plt.legend()
    # plt.show()

    # for i,x in enumerate(testing_rows_x):
    #     output=tanh_perceptron001.predict_output(x)
    #     print(f"Learning rate 1e-2:\noutput:{_reverse_tanh_normalize(output)} - expected:{_reverse_tanh_normalize(testing_rows_y_tanh[i])} - error:{tanh_perceptron001.calculate_error(testing_rows_y_tanh[i],output)}")

    #     output=tanh_perceptron0001.predict_output(x)
    #     print(f"Learning rate 1e-3:\noutput:{_reverse_tanh_normalize(output)} - expected:{_reverse_tanh_normalize(testing_rows_y_tanh[i])} - error:{tanh_perceptron0001.calculate_error(testing_rows_y_tanh[i],output)}")

    #     output=tanh_perceptron00001.predict_output(x)
    #     print(f"Learning rate 1e-4:\noutput:{_reverse_tanh_normalize(output)} - expected:{_reverse_tanh_normalize(testing_rows_y_tanh[i])} - error:{tanh_perceptron00001.calculate_error(testing_rows_y_tanh[i],output)}")

    #     array1.append(tanh_training_error_history1[-1])
    #     array01.append(tanh_training_error_history2[-1])
    #     array001.append(tanh_training_error_history[-1])
    #     array0001.append(tanh_training_error_history3[-1])
    #     array00001.append(tanh_training_error_history4[-1])


    # print(f"Average error after {epochs} epochs for learning rate 1: {np.mean(array1)}")
    # print(f"Average error after {epochs} epochs for learning rate 0.1: {np.mean(array01)}")
    # print(f"Average error after {epochs} epochs for learning rate 1e-2: {np.mean(array001)}")
    # print(f"Average error after {epochs} epochs for learning rate 1e-3: {np.mean(array0001)}")
    # print(f"Average error after {epochs} epochs for learning rate 1e-4: {np.mean(array00001)}")

    logistic_perceptron1: NonLinearPerceptron = NonLinearPerceptron(1,training_rows_x_logistic,training_rows_y_logistic,activation_function=logistic,activation_function_derivate=dlogistic)
    logistic_perceptron01: NonLinearPerceptron = NonLinearPerceptron(0.1,training_rows_x_logistic,training_rows_y_logistic,activation_function=logistic,activation_function_derivate=dlogistic)
    logistic_perceptron001: NonLinearPerceptron = NonLinearPerceptron(1e-2,training_rows_x_logistic,training_rows_y_logistic,activation_function=logistic,activation_function_derivate=dlogistic)
    logistic_perceptron0001: NonLinearPerceptron = NonLinearPerceptron(1e-3,training_rows_x_logistic,training_rows_y_logistic,activation_function=logistic,activation_function_derivate=dlogistic)
    logistic_perceptron00001: NonLinearPerceptron = NonLinearPerceptron(1e-4,training_rows_x_logistic,training_rows_y_logistic,activation_function=logistic,activation_function_derivate=dlogistic)

    logistic_perceptron1.weights = weights.copy()
    logistic_perceptron01.weights = weights.copy()
    logistic_perceptron001.weights = weights.copy()
    logistic_perceptron0001.weights = weights.copy()
    logistic_perceptron00001.weights = weights.copy()

    logistic_training_error_history1, logistic_convergence1 = logistic_perceptron1.train_perceptron(epochs=epochs,epsilon=epsilon)
    logistic_training_error_history01, logistic_convergence01 = logistic_perceptron01.train_perceptron(epochs=epochs,epsilon=epsilon)
    logistic_training_error_history001, logistic_convergence001 = logistic_perceptron001.train_perceptron(epochs=epochs,epsilon=epsilon)
    logistic_training_error_history0001, logistic_convergence0001 = logistic_perceptron0001.train_perceptron(epochs=epochs,epsilon=epsilon)
    logistic_training_error_history00001, logistic_convergence00001 = logistic_perceptron00001.train_perceptron(epochs=epochs,epsilon=epsilon)

    print(f"Error after {epochs} epochs for logistic perceptron with learning rate 1: {logistic_training_error_history1[-1]}")
    print(f"Error after {epochs} epochs for logistic perceptron with learning rate 0.1: {logistic_training_error_history01[-1]}")
    print(f"Error after {epochs} epochs for logistic perceptron with learning rate 1e-2: {logistic_training_error_history001[-1]}")
    print(f"Error after {epochs} epochs for logistic perceptron with learning rate 1e-3: {logistic_training_error_history0001[-1]}")
    print(f"Error after {epochs} epochs for logistic perceptron with learning rate 1e-4: {logistic_training_error_history00001[-1]}")

    plt.plot(logistic_training_error_history1, label=r'$\eta = 1$')
    plt.plot(logistic_training_error_history01, label=r'$\eta = 0.1$')
    plt.plot(logistic_training_error_history001, label=r'$\eta = 1e-2$')
    plt.plot(logistic_training_error_history0001, label=r'$\eta = 1e-3$')
    plt.plot(logistic_training_error_history00001, label=r'$\eta = 1e-4$')

    plt.title("Logistic function with varying learning rates")
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # plt.plot(tanh_training_error_history1, label='lr = 1')
    # plt.plot(tanh_training_error_history2, label='lr = 0.1')
    # plt.plot(tanh_training_error_history, label='lr = 1e-2')
    # plt.plot(tanh_training_error_history3, label='lr = 1e-3')
    # plt.plot(tanh_training_error_history4, label='lr = 1e-4')

    # nlteh2 = non_linear_perceptron2.train_perceptron(epochs=epochs,epsilon=epsilon)
    # nlteh3 = non_linear_perceptron3.train_perceptron(epochs=epochs,epsilon=epsilon)
    # nlteh4 = non_linear_perceptron4.train_perceptron(epochs=epochs,epsilon=epsilon)

    # lteh2 = linear_perceptron2.train_perceptron(epochs=epochs,epsilon=epsilon)
    # lteh3 = linear_perceptron3.train_perceptron(epochs=epochs,epsilon=epsilon)
    # lteh4 = linear_perceptron4.train_perceptron(epochs=epochs,epsilon=epsilon)

    # non_linear_average = np.mean([non_linear_training_error_history, nlteh2, nlteh3, nlteh4], axis=0)
    # linear_average = np.mean([linear_training_error_history, lteh2, lteh3, lteh4], axis=0)
    
    # plt.plot(non_linear_average, label='Non Linear')

    # ALL FUNCTIONS
    # print(f"softplus: {softplus_training_error_history}")
    # print(f"relu: {relu_training_error_history}")
    # print(f"linear: {linear_training_error_history}")

    # plt.plot(linear_training_error_history, label='Linear')
    # print(f"Error after {epochs} epochs for linear perceptron: {linear_training_error_history[-1]}")

    # plt.plot(tanh_training_error_history, label='Tanh')
    # plt.plot(logistic_training_error_history, label='Logistic')
    # plt.plot(relu_training_error_history, label='ReLU')
    # plt.plot(softplus_training_error_history, label='Softplus')

    # plt.plot(non_linear_training_error_history, label='Non Linear')
    # plt.plot(nlteh2, label='Non Linear 2')
    # plt.plot(nlteh3, label='Non Linear 3')
    # plt.plot(nlteh4, label='Non Linear 4')

    # plt.ylim(0, 10)
    # plt.xlabel('Epochs')
    # plt.ylabel('Error')
    # plt.legend(loc='upper right')
    # plt.show()

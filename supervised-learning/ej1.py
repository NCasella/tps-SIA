from src.simple_perceptron import SimplePerceptron

if __name__=="__main__":
    dataset = ((-1,-1), (-1, 1), (1,-1), (1,1))
    and_output = (-1, -1, -1, 1)
    xor_output = (-1, 1, 1, -1)
    perc=SimplePerceptron(5e-2,dataset,and_output)
    perc.train_perceptron(1000,2e-2)
    print(perc.predict_output((-1, -1)))
    print(perc.predict_output((-1, 1)))
    print(perc.predict_output((1, -1)))
    print(perc.predict_output((1, 1)))
    

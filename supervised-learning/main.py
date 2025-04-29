from src.simple_perceptron import SimplePerceptron

if __name__=="__main__":
    dataset = ((-1,-1), (-1, 1), (1,-1), (1,1))
    and_output = (-1, -1, -1, 1)
    xor_output = (-1, 1, 1, -1)
    perc=SimplePerceptron(1e-3,dataset,xor_output)
    perc.train_perceptron(100000,0.005)
    print(perc.predict_output((-1, -1)))
    print(perc.predict_output((-1, 1)))
    print(perc.predict_output((1, -1)))
    print(perc.predict_output((1, 1)))
    

import numpy as np 
import pandas as pd 
import sys
import json 
from src.hopfield import Hopfield
import matplotlib.pyplot as plt

def read_number_files(file,block_size=25):
    with open(file, 'r') as f:
        text = f.read() 
    numbers = text.strip().split()  
    numbers = [int(num) for num in numbers]  
    blocks = [numbers[i:i+block_size] for i in range(0, len(numbers), block_size)]  
    blocks = [block for block in blocks if len(block) == block_size]    
    return blocks

def add_noise_to_dataset(dataset, noise_level):
    noisy_dataset = []
    for sample in dataset:
        noisy_sample = []
        for value in sample:
            if np.random.rand() < noise_level:
                noisy_sample.append(-1 if value==1 else 1) 
            else:
                noisy_sample.append(value)
        noisy_dataset.append(noisy_sample)
    return noisy_dataset

if __name__=="__main__":
    with open(sys.argv[1],"r") as f:
        config=json.load(f)
    
    letters=read_number_files(config["data_source"])
    iterations=config["iterations"]
    noise_rate=config["noise"]

    hopfield:Hopfield = Hopfield(letters)
    letter_states=[]
    energy_per_input=[]
    matches_amount=[[0 for _ in range(11)] for _ in range(len(letters))]

    # noise_rate = 0.0
    # for t in range(11):
    #     for _ in range(500):
    #         noisy_letters=add_noise_to_dataset(letters,noise_rate)
    #         for i, letter in enumerate(noisy_letters):
    #             states,energy_history=hopfield.predic_output(letter,iterations)
    #             matches_amount[i][t] += 1 if np.array_equal(states[-1], letters[i]) else 0
    #     noise_rate += 0.05
            
    # print("Matches amount:", matches_amount)
    x_amount = [0, 0]
    reverse_x = np.array(letters[0]) * (-1)

    noise_rate = 0.5
    for _ in range(500):
        noisy_letters=add_noise_to_dataset(letters,noise_rate)
        for i, letter in enumerate(noisy_letters):
            states,energy_history=hopfield.predic_output(letter,iterations)
            x_amount[0] += 1 if np.array_equal(states[-1], letters[0]) else 0
            x_amount[1] += 1 if np.array_equal(states[-1], reverse_x) else 0

    print("X amount:", x_amount)

    # noisy_letters=add_noise_to_dataset(letters,noise_rate)
    # states,energy_history=hopfield.predic_output(noisy_letters[0],iterations)
    # for i in range(len(states)):
    #     plt.matshow(np.array(states[i]).reshape(5,5), fignum=None, cmap='gray_r')
    #     plt.savefig(f"state_{i}.png")

    # matches_amount = [[200, 200, 197, 177, 168, 147, 131, 101, 80, 59, 28], [200, 200, 195, 191, 176, 173, 141, 139, 102, 76, 57], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [200, 121, 116, 94, 73, 60, 45, 25, 17, 9, 2]]
    # x = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    # for i in range(len(matches_amount)):
    #     matches_amount[i] = [x / 5 for x in matches_amount[i]]

    # plt.plot(x, matches_amount[0], label="Letter X", marker='o')
    # plt.plot(x, matches_amount[1], label="Letter L", marker='o')
    # plt.plot(x, matches_amount[2], label="Letter O", marker='o')
    # plt.plot(x, matches_amount[3], label="Letter T", marker='o')
    # plt.xlabel("Noise Rate")
    # plt.ylabel("Amount of correct matches (%)")
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    # plt.savefig("hopfield_noise.png")

    # plt.plot(range(len(energy_history)), energy_history, label="Energy")
    # plt.xlabel("Epochs")
    # plt.ylabel("Energy")
    # plt.savefig("energy_plot.png")

import json
import sys
from src.kohonen import Kohonen
from src.similarity_metrics import get_metric_function
import pandas as pd

if __name__=="__main__":
    
    with open(sys.argv[1],"r") as f:
        config=json.load(f)

    grid_size=config["grid_size"]
    df=pd.read_csv(config["data_source"],delimiter=',')
    radius=config["radius"]
    constant_radius=config["constant_radius"]
    learning_rate=config["learning_rate"]

    sim_function=get_metric_function(config["similarity_metric"])
    iterations=config["iterations"]

    data=df.drop(columns=["Country"]).to_numpy()
    
    kohonen: Kohonen=Kohonen(grid_size, data, sim_function,radius, constant_radius)
    kohonen.train_network(iterations=iterations)


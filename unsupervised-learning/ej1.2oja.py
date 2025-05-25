import pandas as pd
import numpy as np 
from src.oja import Oja
import json
import sys

if __name__=="__main__":
    with open(sys.argv[1],"r") as f:
        config=json.load(f)
    
    df=pd.read_csv(config["data_source"],delimiter=',')
    
    data=df.drop(columns=["Country"])
    standarized_data=(data-data.mean())/data.std()

    iterations=config["iterations"]
    learning_rate=config["learning_rate"]
    constant_learning_rate=config["constant_learning_rate"]



    oja: Oja=Oja(standarized_data,learning_rate,constant_learning_rate)

    
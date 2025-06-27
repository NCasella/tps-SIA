import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.oja import Oja
import json
import sys

if __name__=="__main__":
    with open(sys.argv[1],"r") as f:
        config=json.load(f)
    
    df=pd.read_csv(config["data_source"],delimiter=',')
    
    data=df.drop(columns=["Country"])
    standarized_data=(data-data.mean())/data.std(ddof=0)

    iterations=config["iterations"]
    learning_rate=config["learning_rate"]
    constant_learning_rate=config["constant_learning_rate"]


    oja: Oja=Oja(standarized_data,learning_rate,constant_learning_rate)
    norm= oja.train_network(iterations)
    result=oja.map_input(standarized_data)
    for i in range(len(norm)):
        print(f"{data.columns[i]}:{norm[i]}")
    print("------------------------------")
    for i in range(len(result)):
        country=df["Country"][i]
        print(f"{country}: {result[i]}")

    feature_norm_pairs = list(zip(data.columns[:len(norm)], norm))

    # hardcoded values from running pca.py
    pc1_values_dict = {
        'GDP': 0.500506,
        'Life.expect': 0.482873,
        'Pop.growth': 0.475704,
        'Inflation': -0.406518,
        'Unemployment': -0.271656,
        'Military': -0.188112,
        'Area': -0.124874
    }

    features_norm_pairs = list(zip(data.columns[:len(norm)], norm))

    merged_data = [
        (feature, norm_value, pc1_values_dict.get(feature, 0))
        for feature, norm_value in features_norm_pairs
    ]

    merged_data.sort(key=lambda x: abs(x[1]), reverse=True)

    features_sorted, norm_sorted, pc1_sorted = zip(*merged_data)

    x = np.arange(len(features_sorted))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, norm_sorted, width, label='Oja', color='skyblue')

    plt.xlabel('Features')
    plt.ylabel('Values')
    plt.grid(axis="y")
    plt.title('Oja primary component coefficients')
    plt.xticks(x, features_sorted, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig("oja-no-pca.png")

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, norm_sorted, width, label='Oja', color='skyblue')
    plt.bar(x + width / 2, pc1_sorted, width, label='PCA', color='orange')

    plt.xlabel('Features')
    plt.ylabel('Values')
    plt.grid(axis="y")
    plt.title('Comparison of Oja vs PCA')
    plt.xticks(x, features_sorted, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pca-vs-oja.png")


    country_result_pairs = list(zip(df["Country"][:len(result)], result))

    country_result_pairs.sort(key=lambda x: x[1], reverse=True)  # descending

    sorted_countries, sorted_results = zip(*country_result_pairs)

    x = np.arange(len(sorted_countries))

    plt.figure(figsize=(14, 6))
    plt.bar(x, sorted_results, color='cornflowerblue')
    plt.xticks(x, sorted_countries, rotation=90)
    plt.xlabel('Country')
    plt.ylabel('Value')
    plt.title('Country Results (Sorted by Value)')
    plt.tight_layout()
    plt.savefig("oja-country-scores.png")

    
    
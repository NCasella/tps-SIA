import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

if __name__=="__main__":
    df = pd.read_csv(sys.argv[1])


    X = df.drop(columns=['Country'])


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2) 
    X_pca = pca.fit_transform(X_scaled)

    # Explained variance
    print("Explained variance ratio:", pca.explained_variance_ratio_)

    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_df['Country'] = df['Country']


    plt.figure(figsize=(8,6))
    plt.scatter(pca_df['PC1'], pca_df['PC2'])
    for i, country in enumerate(pca_df['Country']):
        plt.annotate(country, (pca_df['PC1'][i], pca_df['PC2'][i]))
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Countries')
    plt.grid(True)
    plt.savefig("res.png")
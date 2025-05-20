import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def biplot(score, coeff, labels=None):
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]
    plt.scatter(xs, ys, alpha=0.5)
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0]*2, coeff[i, 1]*2, color='r', alpha=0.5, head_width=0.05)
        if labels is None:
            plt.text(coeff[i, 0]*2.2, coeff[i, 1]*2.2, "Var"+str(i+1), color='r')
        else:
            plt.text(coeff[i, 0]*2.2, coeff[i, 1]*2.2, labels[i], color='r')

if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1])
    X = df.drop(columns=['Country'])

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    print("Explained variance ratio:", pca.explained_variance_ratio_)

    # Create PCA DataFrame
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_df['Country'] = df['Country']

    # --- PCA Scatter Plot with Annotations ---
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_df['PC1'], pca_df['PC2'])
    for i, country in enumerate(pca_df['Country']):
        plt.annotate(country, (pca_df['PC1'][i], pca_df['PC2'][i]), fontsize=8)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Countries')
    plt.grid(True)
    plt.savefig("res_scatter.png")
    plt.close()

    # --- Biplot ---
    plt.figure(figsize=(8, 6))
    biplot(X_pca, pca.components_.T, labels=X.columns)
    for i, country in enumerate(pca_df['Country']):
        plt.annotate(country, (pca_df['PC1'][i], pca_df['PC2'][i]), fontsize=8)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Biplot')
    plt.grid(True)
    plt.savefig("res_biplot.png")
    plt.close()

    # --- Boxplot of Standardized Features ---
    standardized_df = pd.DataFrame(X_scaled, columns=X.columns)
    standardized_df['Country'] = df['Country']

    plt.figure(figsize=(12, 6))
    standardized_df.drop(columns=['Country']).boxplot(rot=45)
    plt.title("Boxplot of Standardized Features")
    plt.ylabel("Standardized Value")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("res_boxplot.png")
    plt.close()

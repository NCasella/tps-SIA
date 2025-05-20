import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import logging

def biplot(score, coeff, labels=None):
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]
    plt.scatter(xs, ys, alpha=0.5)
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0] * 2, coeff[i, 1] * 2, color="r", alpha=0.5, head_width=0.05)
        if labels is None:
            plt.text(coeff[i, 0] * 2.2, coeff[i, 1] * 2.2, "Var" + str(i + 1), color="r")
        else:
            plt.text(coeff[i, 0] * 2.2, coeff[i, 1] * 2.2, labels[i], color="r")

if __name__ == "__main__":

    logger = logging.getLogger("pca.py")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(asctime)s - %(filename)s:%(lineno)s: %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")

    df = pd.read_csv(sys.argv[1])
    X = df.drop(columns=["Country"])

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)


    output = "output"
    logger.info(f"Using '%s' as the output directory", output)

    # Create PCA DataFrame
    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    pca_df["Country"] = df["Country"]

    os.makedirs("output", exist_ok=True)

    # Explained variance
    with open(f"{output}/explained_variance_ratio.txt", "w") as f:
        for i in range(pca.explained_variance_ratio_.shape[0]):
            f.write(f"PC{i+1}: {str(pca.explained_variance_ratio_[i])}\n")

    # --- Biplot ---
    plt.figure(figsize=(8, 6))
    biplot(X_pca, pca.components_.T, labels=X.columns)
    for i, country in enumerate(pca_df["Country"]):
        plt.annotate(country, (pca_df["PC1"][i], pca_df["PC2"][i]), fontsize=8)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Biplot")
    plt.grid(True)
    plt.savefig(f"{output}/res_biplot.png")
    plt.close()

    # --- Boxplot of Standardized Features ---
    standardized_df = pd.DataFrame(X_scaled, columns=X.columns)
    standardized_df["Country"] = df["Country"]

    plt.figure(figsize=(12, 6))
    standardized_df.drop(columns=["Country"]).boxplot(rot=45)
    plt.title("Boxplot of Standardized Features")
    plt.ylabel("Standardized Value")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output}/res_boxplot.png")
    plt.close()

    plt.figure(figsize=(12, 6))

    # Countries sorted by PC1
    sorted_df = pca_df.sort_values(by="PC1", ascending=False).drop(columns="PC2")
    with open(f"{output}/countries_by_pc1.txt", "w") as f:
        f.write(str(sorted_df))

    # PC1 Scores
    plt.bar(sorted_df["Country"], sorted_df["PC1"], color="skyblue")
    plt.xticks(rotation=90)
    plt.ylabel("PC1 Score")
    plt.title("Countries Projected on First Principal Component (PC1)")
    plt.tight_layout()
    plt.grid(axis="y")
    plt.savefig(f"{output}/pc1_scores.png")

    plt.figure(figsize=(12, 6))

    # Countries sorted by PC2
    sorted_df = pca_df.sort_values(by="PC2", ascending=False).drop(columns="PC1")
    with open(f"{output}/countries_by_pc2.txt", "w") as f:
        f.write(str(sorted_df))

    # PC2 Scores
    plt.bar(sorted_df["Country"], sorted_df["PC2"], color="skyblue")
    plt.xticks(rotation=90)
    plt.ylabel("PC1 Score")
    plt.title("Countries Projected on Second Principal Component (PC2)")
    plt.tight_layout()
    plt.grid(axis="y")
    plt.savefig(f"{output}/pc2_scores.png")

    loadings = pd.DataFrame(
        pca.components_.T,
        index=X.columns,
        columns=["PC1", "PC2"]
    )

    # Sorted coefficients of each one
    with open(f"{output}/pc1_coefficients.txt", "w") as f:
        f.write(str(loadings.sort_values(by="PC1", ascending=False, key=abs).drop(columns="PC2")))
    with open(f"{output}/pc2_coefficients.txt", "w") as f:
        f.write(str(loadings.sort_values(by="PC2", ascending=False, key=abs).drop(columns="PC1")))

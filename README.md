# Gaussian Mixture Models (GMM) vs. K-Means

## Introduction  
This project compares **K-Means** and **Gaussian Mixture Models (GMM)** for clustering using the **Iris dataset**. It explores their theoretical differences, implementation, visualization, and evaluation using the **Silhouette Score**.  

## Features  
- Implementation of K-Means and GMM in Python  
- Visualization of clustering results  
- Evaluation using Silhouette Score  
- Comparison of strengths and weaknesses  

## Technologies Used  
- Python (3.x)  
- Scikit-learn  
- NumPy  
- Matplotlib & Seaborn  

## Dataset  
- Name: Iris Dataset  
- Samples: 150  
- Features: Sepal & Petal length/width  
- Classes: Setosa, Versicolor, Virginica  

## Installation  
1. Clone the repository:  
  bash
   git clone - Clone the repo
   cd Gaussian Mixture Models.ipynb
  
2. Install dependencies:  
  bash
 pip install numpy matplotlib Seaborn 

4. Run the script:  
  bash
   Gaussian Mixture Models.ipynb
   

## Usage  
- Run the script to perform clustering on the Iris dataset  
- Compare K-Means and GMM using visualizations and metrics  

## Implementation  

### Load the Data  

from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
X = iris.data  # Features
y = iris.target  # True labels
```

### Apply K-Means Clustering  

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X)


### Apply GMM Clustering  

from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3, random_state=42)
gmm_labels = gmm.fit_predict(X)


### Visualization  

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 5))

# K-Means Clustering
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis')
plt.title("K-Means Clustering")

# GMM Clustering
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=gmm_labels, cmap='viridis')
plt.title("GMM Clustering")

plt.show()


Evaluation using Silhouette Score  

from sklearn.metrics import silhouette_score

kmeans_score = silhouette_score(X, kmeans_labels)
gmm_score = silhouette_score(X, gmm_labels)

print(f"K-Means Silhouette Score: {kmeans_score}")
print(f"GMM Silhouette Score: {gmm_score}")
```

Results & Evaluation  
| Method   | Assumptions         | Clustering Type | Performance |  
|----------|---------------------|----------------|-------------|  
| K-Means  | Spherical clusters  | Hard clustering | Fast but less flexible |  
| GMM      | Elliptical clusters | Soft clustering | More flexible, better for complex data |  

 Conclusion  
 K-Means is simple, efficient, and best suited for spherical clusters.  
 GMM is more flexible, accommodating elliptical clusters and soft assignments.  
The Silhouette Score often favors **GMM** when clusters are not perfectly spherical.  

Future Improvements  
Experiment with different datasets.  
Optimize the number of clusters using **Elbow Method** or **Bayesian Information Criterion (BIC)**.  
Apply **dimensionality reduction techniques (PCA, t-SNE)** to visualize high-dimensional clustering.  


## License  
This project is licensed under the **MIT License**.  

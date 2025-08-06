import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

df=pd.read_csv('cleaned_survey.csv')

df=df[df['tech_company']=='Yes']



features = ['Gender', 'self_employed', 'family_history', 'treatment',
       'work_interfere', 'remote_work', 'benefits', 'care_options',
       'wellness_program', 'seek_help', 'leave', 'mental_health_consequence',
       'coworkers', 'supervisor', 'mental_health_interview'

]

# creating feauture dataset 
X= df[features]


cat_cols = X.select_dtypes(include = 'object').columns.tolist()
num_cols = X.select_dtypes(include = ['int64','float64']).columns.tolist()



preprocessing = ColumnTransformer([('cat',OneHotEncoder(sparse_output=False, handle_unknown='ignore'),cat_cols),('num',StandardScaler(),num_cols)],remainder='passthrough')


X_transformed = preprocessing.fit_transform(X)

# Reducing dimensionality to 4
X_pca = PCA(n_components=4).fit_transform(X_transformed)

# using tsne with best n_components
tsne = TSNE(n_components=2, perplexity=40, learning_rate=200, max_iter=2000, init='pca', random_state=42) 
X_embedded = tsne.fit_transform(X_pca)
  
#using kmeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_embedded)
    
  
#evaluating
score = silhouette_score(X_embedded, labels)
print("Silhouette Score:", score)


#PLotting clusters
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap='Set2', s=50, alpha=0.7)
plt.title("t-SNE Visualization of Clusters")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.colorbar(scatter, label="Cluster")
plt.grid(True)
plt.tight_layout()
plt.show()




import pandas as pd

df = pd.read_csv('Final_report1 (1).csv')
df.head()

df.shape

x = df['Title']
x[0]
x.head()



import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

wordnet = PorterStemmer()       #"""for limmatization"""
# sentences = nltk.sent_tokenize(x)
corpus = []
print(x[0])

for i in x:
    a = re.sub('[^a-zA-Z]', ' ', i)
    a = a.lower()
    a = a.split()
    a = [wordnet.stem(word) for word in a if not word in set(stopwords.words('english'))]   #"""for lemmatization"""
    a = ' '.join(a)
    corpus.append(a) 
print(corpus[0])
# df['corpus'] = corpus
df.head()

from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features=200000)
vector = cv.fit_transform(corpus).toarray()
vector
vector.shape

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
    
sse = []
for k in range(2,20,2):
    sse.append(KMeans(n_clusters=k).fit(vector).inertia_)
    print('Fit {} clusters'.format(k))
        
f, ax = plt.subplots(1, 1)
ax.plot(range(2,20,2), sse, marker='o')
ax.set_xlabel('Cluster Centers')
ax.set_xticks(range(2,20,2))
ax.set_xticklabels(range(2,20,2))
ax.set_ylabel('SSE')
ax.set_title('SSE by Cluster Center Plot')
plt.show()






from sklearn.cluster import KMeans
km = KMeans(10)
clusters=km.fit_predict(vector)
clusters = pd.DataFrame(clusters)
clusters

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# pca = PCA(n_components=2)
# reduced_features = pca.fit_transform(vector)# reduce the cluster centers to 2D
# reduced_cluster_centers = pca.transform(kmeans.cluster_centers_)
# reduced_features=pd.DataFrame(reduced_features)
# reduced_features=reduced_features.rename(columns={0:'Feature1',1:'Feature2'})
# reduced_features    
# clusters = clusters.rename(columns={0:'cluster'})
# Final_Report=pd.concat([reduced_features,clusters],axis=1)
# Final_Report
# import seaborn as sns
# plt.figure(figsize=(12,6))
# sns.scatterplot(y='Feature1',x='Feature2',data=Final_Report,hue='Clusters',palette='gist_rainbow')
# plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=150, c='b')
# plt.legend()
# plt.show()


# from sklearn.metrics import silhouette_score
# silhouette_score(vector, labels=kmeans.predict(vector))


# Final_Report['Clusters'].value_counts()
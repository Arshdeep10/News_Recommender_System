from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
import nltk
import pandas as pd


df = pd.read_csv('total.csv')
df.head()

df.shape

x = df['content']
x[0]
x.head()


wordnet = PorterStemmer()  # """for limmatization"""
# sentences = nltk.sent_tokenize(x)
corpus = []
print(x[0])

for i in x:
    a = re.sub('[^a-zA-Z]', ' ', str(i))
    a = a.lower()
    a = a.split()
    a = [wordnet.stem(word) for word in a if not word in set(
        stopwords.words('english'))]  # """for lemmatization"""
    a = ' '.join(a)
    corpus.append(a)
print(corpus[0])
# df['corpus'] = corpus
df.head()

cv = TfidfVectorizer(max_features=200000)
vector = cv.fit_transform(corpus).toarray()
vector
vector.shape


#elbow method
sse = []
for k in range(2, 20, 2):
    sse.append(KMeans(n_clusters=k).fit(vector).inertia_)
    print('Fit {} clusters'.format(k))

f, ax = plt.subplots(1, 1)
ax.plot(range(2, 20, 2), sse, marker='o')
ax.set_xlabel('Cluster Centers')
ax.set_xticks(range(2, 20, 2))
ax.set_xticklabels(range(2, 20, 2))
ax.set_ylabel('SSE')
ax.set_title('SSE by Cluster Center Plot')
plt.show()


Silhouette_score=[]
for i in range(2,20):
    kmeans=KMeans(n_clusters=i,random_state=0)
    sil_score=silhouette_score(vector, labels=kmeans.fit_predict(vector)) 
    Silhouette_score.append(sil_score)
print(Silhouette_score)
number_of_Clusters=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
plt.plot(range(2,21),Silhouette_score) 
plt.title("The Silhouette_Score")
plt.xlabel("number of Clusters")
plt.ylabel("The Silhouette Score")
plt.show()


vocab = vectorizer.get_feature_names()
pd.DataFrame(np.round(x, 2), columns=vocab)

km = KMeans(10, random_state=0)
clusters = km.fit_predict(vector)
clusters = pd.DataFrame(clusters)
clusters
cluster_labels1 = km.labels_

cluster_labels = pd.DataFrame(cluster_labels1, columns=['ClusterLabel'])
cluster_labels


Final_Report=pd.concat([vector,cluster_labels],axis=1)
Final_Report

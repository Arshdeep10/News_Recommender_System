from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
import nltk
import pandas as pd

df = pd.read_csv('total.csv')
df

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

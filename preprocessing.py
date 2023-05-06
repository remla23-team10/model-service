import re
import pickle
from urllib.request import urlopen
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

class Preprocessing:

    def __init__(self):
        self.ps = PorterStemmer()

        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')

        self.stopwords = all_stopwords


    def preprocess_dataset(self, dataset):
        corpus = []

        for i in range(0, dataset.shape[0]):
            corpus.append(self.preprocess_review(dataset['Review'][i]))
        return corpus
        

    def preprocess_review(self, review):

        review = re.sub('[^a-zA-Z]', ' ', review)
        review = review.lower()
        review = review.split()
        review = [self.ps.stem(word) for word in review if not word in set(self.stopwords)]
        review = ' '.join(review)
        return review
    
    def fit_transform(self, corpus):
        self.cv = CountVectorizer(max_features = 1420)
        return self.cv.fit_transform(corpus).toarray()
    
    def transform(self, corpus):
        if self.cv is None:
            raise Exception("Vectorizer is not initialized. Please call fit_transform or load from PKL or URL first.")
        return self.cv.transform(corpus).toarray()
    
    def vectorizer_from_pkl(self, cvFile):
        self.cv = pickle.load(open(cvFile, "rb"))

    def vectorizer_from_url(self, url):
        self.cv = pickle.load(urlopen(url))
    
    def save_vectorizer(self, filename):
        pickle.dump(self.cv, open(filename, "wb"))

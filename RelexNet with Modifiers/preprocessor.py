import nltk
from functools import lru_cache
import re
import string
from nltk.corpus import stopwords

class Preprocessor:
    def clean_text(self, text):
        # '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
        # make text lowercase
        text1 = text.lower()
        # remove square brackets
        text1 = re.sub('\[.*?\]', '', text1)
        #remove <>
        text1 = re.sub('<.*?>+', '', text1)
        text1 = re.sub('\(.*?\)', ' ', text1)
        text1 = re.sub('\{.*?\}', ' ', text1)
        # remove links
        text1 = re.sub('https?://\S+|www\.\S+', ' ', text1)
        # remove punctuation
        text1 = re.sub('[%s]' % re.escape(string.punctuation), '', text1)
        # remove \n
        # text = re.sub('\n', '', text)
        # remove numbers
        text = re.sub('\w*\d\w*', '', text)
        return text1

    def __init__(self):
        self.stem = lru_cache(maxsize=10000)(nltk.stem.SnowballStemmer('english').stem)
        # self.stopwords = stopwords.words('english')
        self.tokenize = nltk.tokenize.TreebankWordTokenizer().tokenize

    def __call__(self, text):
        text1 = self.clean_text(text)
        tokens = self.tokenize(text1)
        # tokens = [token for token in tokens if token not in self.stopwords]
        # tokens = [self.stem(token) for token in tokens]

        return tokens

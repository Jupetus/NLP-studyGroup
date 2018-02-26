import numpy as np
import string
import nltk
from itertools import islice
from collections import Counter
from nltk import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

class ngrams:
    tokens = []
    grams = []
    # Process the data
    def preProcess(self, data):
        # Tokenize, make string to a array
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(data)
        # Kill punctuations
        translator = str.maketrans('','', string.punctuation)
        depunctuated = [token.translate(translator) for token in tokens]
        # Remove stopwords
        # unstoppable = [token for token in depunctuated if token not in stopwords.words('english')]
        # words = [word.lower() for word in unstoppable]
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        lemmatized = [lemmatizer.lemmatize(word) for word in depunctuated]
        # Stem
        stemmer = SnowballStemmer('english')
        stemmed = [stemmer.stem(word) for word in lemmatized]
        return stemmed

    # Calculates a word frequency untill to order n
    def calculate_fw(self, tokens, n=1):
        gramDict = {}
        # Loop through tokens
        for k in range(0, len(tokens) - n + 1):
            tmpWord = " ".join(tokens[k:k + n])
            if tmpWord not in gramDict.keys():
                gramDict[tmpWord] = 1
            else:
                gramDict[tmpWord] += 1
        return gramDict


    # Calculates frequencies for all n-grams untill order n
    def calculate_wf_to_n(self, tokens, order):
        for i in range(order):
            self.grams.append(self.calculate_fw(tokens,i + 1))

    # Returns all frequency distributions
    def get_fd(self, n=1):
        return self.grams[n - 1]

    # Gets word frequency in a specific distribution
    def get_word_fd(self, word, n=1):
        return self.grams[n - 1][word]

    def getTokens(self):
        return self.tokens

    def countdata(self, data):
        return 0

    def __init__(self, data, order=2):
        # Tokenize the words
        self.tokens = self.preProcess(data)
        self.calculate_wf_to_n(self.tokens, order)
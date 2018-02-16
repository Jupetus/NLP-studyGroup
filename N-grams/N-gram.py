import numpy as np
import nltk
from nltk.stem import SnowballStemmer
from nltk.util import ngrams
from nltk import word_tokenize
from collections import Counter
from nltk.collocations import *

def calculatediscountparmas(gram1, gram2):
    y = len(gram1) / (len(gram1) + 2 * len(gram2))
    discount = 1 - 2 * y * len(gram2) / len(gram1)
    return discount

def getngramprob(gram1, gram2, word, suffix, smoothing=""):
    uniqueWords = len(gram2)
    totalWords = sum(unigram_fd.values())
    divident = gram1[(suffix, word)]
    divisor = gram2[suffix]
    # add k-smoothing
    if smoothing == "k":
        return (divident + 1) / (divisor + uniqueWords)
    # discount smoothing
    if smoothing == "d":
        discount = calculatediscountparmas(gram2, gram1)
        if divident != 0:
            return (divident - discount) / divisor
        return discount / divisor
    if smoothing == "bo":
        #TODO: weight values!
        if divident != 0:
            return 0.8 * divident / divisor
        else:
            return 0.2 * gram2[word] / totalWords
    if smoothing == "sbo":
        if divident != 0:
            return divident / divisor
        else:
            return 0.4 * gram2[word] / totalWords
    else:
        return divident / divisor

stemmer = SnowballStemmer('english')
path = "Dataset2.txt"
Reviews = []
ngramList = []
c = Counter()

bigram_measures = nltk.collocations.BigramAssocMeasures()
with open(path) as f:
    bigram_fd = nltk.FreqDist()
    unigram_fd = nltk.FreqDist()

    for line in f.readlines():
        review = line.split('$')
        tokenized = word_tokenize(review[2].lower())

        finder = BigramCollocationFinder.from_words(tokenized)
        bigram_fd += nltk.FreqDist(finder.ngram_fd)
        unigram_fd += nltk.FreqDist(finder.word_fd)

    finder = BigramCollocationFinder(unigram_fd, bigram_fd)
    pmi_scores = finder.score_ngrams(bigram_measures.pmi)
    t_score = finder.score_ngrams(bigram_measures.student_t)

print(unigram_fd["the"] / 79000)
print(getngramprob(bigram_fd, unigram_fd, "the", "cat", smoothing="sbo"))
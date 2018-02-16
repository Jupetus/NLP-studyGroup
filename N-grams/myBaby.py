import numpy as np
import nltk
from nltk.stem import SnowballStemmer
from nltk.util import ngrams
from nltk import word_tokenize
from collections import Counter
from nltk.collocations import *

stemmer = SnowballStemmer('english')

def stemMyStrings(words):
    newListSAS = []
    for word in words:
        newListSAS.append(stemmer.stem(word))

    return "".join(newListSAS)

def makeGram(strings, n):
    # Loop all words that are fiven as input
    gramList = []
    gramDict = {}
    for string in strings:
        split = string.split()
        for k in range(0, len(split) - n + 1):
            tmpWord  = " ".join(split[k:k+n])
            if tmpWord not in gramDict.keys():
                gramDict[tmpWord] = 1
            else:
                gramDict[tmpWord] += 1
            gramList.append(" ".join(split[k:k+n]))


    #n_gram = countGram(gramList)
    return gramDict

def ngramProbability(ngram, nminusgram, word, suffix, smoothing = "", k = 1):
    if suffix + ' ' + word in ngram.keys():

    if smoothing == "k":
        words = len(nminusgram)
        return (ngram[suffix + ' ' + word] + k) / (nminusgram[suffix] + k * words)
    else:
        if suffix + ' ' + word in ngram.keys():
            return ngram[suffix + ' ' + word] / nminusgram[suffix]
    return 0

def calculatePMI(jointDist, soloDist, word1, word2):
    if word1 + ' ' + word2 in jointDist.keys():
        return np.log2(float(jointDist[word1 + ' ' + word2] / (soloDist[word1] * soloDist[word2])))
    return 0


stemmer = SnowballStemmer('english')
path = "Dataset2.txt"
Reviews = []
ngramList = []
c = Counter()
with open(path) as f:
    for line in f.readlines():
        review = line.split('$')
        tokenized = word_tokenize(review[2].lower())
        ngramList.append(stemMyStrings(" ".join(tokenized)))



bigram = makeGram(ngramList, 2)
unigram = makeGram(ngramList, 1)

#test = sorted(bigram.items(), key=lambda x: x[1], reverse=True)
# print(test)
# print(bigram["my songs"])
# print(unigram["my"])
# print(unigram["songs"])
# print(calculatePMI(bigram, unigram, "my", "songs"))

print(len(unigram))
print(ngramProbability(bigram, unigram, "cat", "my", smoothing="k"))
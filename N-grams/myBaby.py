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

# Make a n-gram out of list of strings, where each string is a sentance
# Strings = array of strings
# N = order of n-gram
def make_ngram(strings, n=1):
    gramDict = {}
    # Loop strings
    for string in strings:
        split = string.split()
        # Loop substrings of length n and append to dictionary
        for k in range(0, len(split) - n + 1):
            tmpWord  = " ".join(split[k:k+n])
            if tmpWord not in gramDict.keys():
                gramDict[tmpWord] = 1
            else:
                gramDict[tmpWord] += 1
    return gramDict

# Evaluates given unigram as count(word) / count(*)
def eval_unigram(ngram, word):
    wordcount = sum(ngram.values())
    if word in ngram.keys():
        return ngram[word] / wordcount
    else:
        return 0

# Evaluates given n-gram as: n-gram_count(string) / (n-1)-gram_count(string - 1)
def eval_gram(ngram, mgram, suffix, word):
    cond = suffix + ' ' + word

    if cond in ngram.keys():
        divident = ngram[cond]
        divisor = mgram[suffix]
        return divident / divisor
    else:
        return 0

# Evaluates n-gram with given smoothing parameters
# n-grams: List of n-grams in decreasing order
# word: word we are evaluating
# suffix: string trailing the word
def eval_ngram(ngrams, suffix, word, smoothing = "", k = 1):

    assert len(ngrams) >= 2, "Atleast two grams yo"
    # add k-smoothing
    if smoothing == "k":
        return 1
    # discount smoothing
    if smoothing == "d":
        return 0
    # BackOff smoothing
    if smoothing == "bo":
        return 0
    # Stupid BackOff
    if smoothing == "sbo":
        return 0
    # Normal n-gram
    else:
        return eval_gram(ngrams[0], ngrams[1], suffix, word)


def calculatePMI(ngram2, ngram1, word1, word2):
    if word1 + ' ' + word2 in ngram2.keys():
        divident = eval_gram(ngram2, ngram1, word1, word2)
        divisor = eval_unigram(ngram1, word1) * eval_unigram(ngram1, word2)

        return np.log2(divident / divisor)
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



bigram = make_ngram(ngramList, 2)
unigram = make_ngram(ngramList, 1)
grams = [bigram, unigram]

print(calculatePMI(bigram, unigram, "my", "songs"))

"""
#test = sorted(bigram.items(), key=lambda x: x[1], reverse=True)
# print(test)
# print(bigram["my songs"])
# print(unigram["my"])
# print(unigram["songs"])
# print(calculatePMI(bigram, unigram, "my", "songs"))
print(len(unigram))
print(ngramProbability(bigram, unigram, "cat", "my", smoothing="k"))
"""
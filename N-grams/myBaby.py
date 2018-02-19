import numpy as np
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams
from nltk import word_tokenize
from collections import Counter
from nltk.collocations import *
from operator import itemgetter

"""
===================
TODO: Make it so that works for all grams
- Perplexity
- Some of the smoothings
Gosh this came out looking bit confusing
===================
"""
stemmer = PorterStemmer() # SnowballStemmer('english')
rTokenizer = RegexpTokenizer('\w+')

def stemMyStrings(words):
    newListSAS = []
    for word in words:
        newListSAS.append(stemmer.stem(word))
    return "".join(newListSAS)

def calculate_perplexity(ngrams, n, test_set):
    prod = 1
    N = 0
    for line in test_set:
        split = line.split()
        for k in range(n, len(split)):
            suffix = split[k - 1]
            word = split[k]
            likelihood = eval_ngram_word(ngrams, suffix, word)
            prod *= 1 / likelihood
            N += 1
    return abs(prod) ** (1.0 / N)

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
def eval_unigram(ngram, word, alpha=1.0):
    wordcount = sum(ngram.values())
    if word in ngram.keys():
        return alpha* ngram[word] / wordcount
    else:
        return 0

# Evaluates given n-gram as: n-gram_count(string) / (n-1)-gram_count(string - 1)
def eval_gram(ngram, mgram, suffix, word, k=0, alpha=1.0, wordcount = 1):
    cond = suffix + ' ' + word

    if cond in ngram.keys():
        divident = ngram[cond]
        divisor = mgram[suffix]
        return (divident + k) / (divisor + k * wordcount)
    else:
        return k / wordcount

def kneser_Ney(grams, suffix, word, d=0.5):
    cond = suffix + ' ' + word
    divident = 0
    divisor = grams[1][suffix]
    if cond in grams[0].keys():
        divident = max(grams[0][cond] - d, 0)

    norm_constant = d / divisor
    return divident / divisor + norm_constant * eval_unigram(grams[1], word)

# Evaluates n-gram with given smoothing parameters
# n-grams: List of n-grams in decreasing order
# word: word we are evaluating
# suffix: string trailing the word
def eval_ngram_word(ngrams, suffix, word, smoothing = "", k = 1, alpha=1.0):

    # add k-smoothing
    if smoothing == "k":
        assert len(ngrams) >= 2, "Atleast two grams yo"

        totalWords = sum(ngrams[len(ngrams) - 1].values())
        return eval_gram(ngrams[0], ngrams[1], suffix, word, k=k, wordcount=totalWords)
    # discount smoothing got lazy so like kneser-ney
    if smoothing == "d":
        return kneser_Ney(ngrams, suffix, word)
    # BackOff smoothing
    if smoothing == "bo":
        if suffix + ' ' + word in ngrams[0].keys():
            return alpha * eval_gram(ngrams[0], ngrams[1], suffix, word)
        else:
            if len(ngrams) > 1:
                return eval_ngram_word(ngrams[1:], suffix, word, smoothing=smoothing)
            else:
                return eval_unigram(ngrams[0], word)
    # Stupid BackOff
    if smoothing == "sbo":
        if suffix + ' ' + word in ngrams[0].keys():
            return alpha * eval_gram(ngrams[0], ngrams[1], suffix, word)
        else:
            if len(ngrams) > 1:
                return eval_ngram_word(ngrams[1:], suffix, word, smoothing = smoothing, k = k, alpha=(alpha*0.4))
            else:
                return eval_unigram(ngrams[0], word, alpha=alpha)
    # Normal n-gram
    else:
        assert len(ngrams) >= 2, "Atleast two grams yo"

        return eval_gram(ngrams[0], ngrams[1], suffix, word)

def calculatePMI(ngram2, ngram1, word1, word2):
    if word1 + ' ' + word2 in ngram2.keys():
        divident = eval_gram(ngram2, ngram1, word1, word2)
        divisor = eval_unigram(ngram1, word1) * eval_unigram(ngram1, word2)
        return np.log2(divident / divisor)
    return 0

def find_ml_word(gram2, gram1, word):
    keyList = []
    for key in gram2.keys():
        split = key.split()
        if " ".join(split[:-1]) == word:
            keyList.append(key)

    # Change index next to split according to the n-gram! TODO: make it bettar!
    test = [(key.split()[2], eval_ngram_word([gram2, gram1], word, key.split()[2], smoothing="k")) for key in keyList]
    ml = max(test, key=itemgetter(1))[0]
    ml_word = ml

    return ml_word

def make_a_story(grams, start_word = "i", story_length=100):
    prevString = start_word
    newcond = start_word
    story = start_word.split()
    for i in range(story_length):
        prevString = find_ml_word(grams[0], grams[1], newcond)
        story.append(prevString)
        newcond = " ".join(story[-2:])

    return " ".join(story)



path = "Dataset2.txt"
Reviews = []
ngramList = []
c = Counter()
with open(path) as f:
    for line in f.readlines():
        review = line.split('$')
        tokenized = rTokenizer.tokenize(review[2].lower().strip())
        ngramList.append(stemMyStrings(" ".join(tokenized)))
        review[2] = " ".join(tokenized)
        Reviews.append(review)

print(Reviews[1])
trigram = make_ngram(ngramList, 3)
bigram = make_ngram(ngramList, 2)
unigram = make_ngram(ngramList, 1)
grams = [trigram, bigram, unigram]

# Story part!
print(make_a_story(grams, start_word="i like", story_length=20))

# Evaluate my n-gram
grams = [bigram, unigram]
print(eval_ngram_word(grams, "like", "to", smoothing = "k"))

# PMI calculation
print(calculatePMI(bigram, unigram, "like", "to"))
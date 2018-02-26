import numpy as np
from tools.ngrams import ngrams

def calculate_probability():
    return 0

def classify_word():
    return 0

path = "Dataset2.txt"
reviews = []
pos_reviews = []
neg_reviews = []

with open(path) as f:
    for line in f.readlines():
        review = line.split('$')
        if review[1] == "pos":
            pos_reviews.append(review[2])
        else:
            neg_reviews.append(review[2])
        reviews.append(review[2])

pos_training = pos_reviews[:int(len(pos_reviews)*0.9)]
pos_testing = pos_reviews[int(len(pos_reviews)*0.9) + 1:]
neg_training = pos_reviews[:int(len(neg_reviews)*0.9)]
neg_testing = pos_reviews[int(len(neg_reviews)*0.9) + 1:]

pos_gram = ngrams(" ".join(pos_training), order=1)
neg_grams = ngrams(" ".join(neg_training), order=1)
gram = ngrams(" ".join(reviews))


positive_prior = len(pos_training)
negative_prior = len(neg_training)


print(pos_gram.getTokens())

"""
print(gram.getTokens())
print(gram.get_fd(1))
print(gram.get_word_fd("sound qualiti", 2))
print(gram.get_word_fd("sound", 1))
"""
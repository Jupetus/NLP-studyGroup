import nltk
import numpy as np
from collections import Counter
import re
from nltk import ngrams
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

def Viterbi(observation, state_graph):
    viterbi = np.zeros((len(state_graph) + 2, len(observation)))

    # POS - Part of speech
    # Initialize! t0 <- P(POS) * P(word | POS)


    # t + 1 <- MAX( P(POS_T + 1 | POS_t) * P(word | POS) )

    return 0

def stemMyStrings(words):
    stemmer = PorterStemmer()  # SnowballStemmer('english')
    newListSAS = []
    for word in words:
        newListSAS.append(stemmer.stem(word))
    return "".join(newListSAS)

def load_data():
    rTokenizer = RegexpTokenizer('\w+') #^([^0-9]*)]
    path = "Dataset2.txt"
    Reviews = []
    with open(path) as f:
        for line in f.readlines():
            review = line.split('$')
            tokenized = rTokenizer.tokenize(review[2].strip())
            stemmed = stemMyStrings(" ".join(tokenized))

            review[2] = " ".join(tokenized)
            Reviews.append(re.sub('[0-9]+', '', stemmed))
    return Reviews

# Get corpus with tagged words!
# nltk.download('brown')
brown_data = nltk.corpus.brown.tagged_words()
brown_dictionary = {word: POV for word, POV in brown_data}

def get_bigram_probabilities(tokens):
    bigrams = list(ngrams(tokens, 2))
    unigrams = Counter(tokens)
    probabilities = {}

    for bigram in bigrams:
        denominator = unigrams['NN']
        numerator = bigrams.count(bigram)
        probabilities[bigram] = numerator / denominator
    return probabilities

# Load data and label the words!
all_data = load_data()
complete_data = []
data_size = len(all_data)
data_offset = int(0.8 * data_size)
tokens = []

# Process reviews to Part-of-speech
counter_in_brown = 0
counter_not_brown = 0
for line in all_data:
    tmp = line.split()
    line = []
    for word in tmp:
        if word in brown_dictionary.keys():
            counter_in_brown += 1
            line.append((word, brown_dictionary[word]))
            tokens.append(brown_dictionary[word])
        else:
            counter_not_brown += 1
            line.append((word, "UNK"))
            tokens.append("UNK")
    complete_data.append(line)

# print("In: ", counter_in_brown, " Not: ", counter_not_brown)
# Divide the data!
train_data = complete_data[:data_offset]
test_data = complete_data[data_offset + 1 :]

# Make a trigram from the POS!
# P(t_i | t_{i - 1} ) = C(t_{i - 1}, t_i) / C( t_{i - 1})
probabilities = get_bigram_probabilities(tokens)


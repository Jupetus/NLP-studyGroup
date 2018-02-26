import numpy as np
from collections import Counter

"""

"""

# ================
# Assignment 6.1
# ================
print("================")
print("Assignment 6.1")
print("================")
# Naive Bayes assumption!
# P(f1, f2 ... fn | C) = P(f1 | C) * P(f2 | C) ..... P(fn | C)
p_pos  = np.log2(0.09) + np.log2(0.07) + np.log2(0.29) + np.log2(0.04) + np.log2(0.08)
p_neg = np.log2(0.16) + np.log2(0.06) +  np.log2(0.06) + np.log2(0.15) + np.log2(0.11)

#print("Pos: ", p_pos , " Neg: ", p_neg)
if p_pos > p_neg:
    print("This is positive!")
else:
    print("This is negative!")


# More probable class has higher value!

# ================
# Assignment 6.2
# ================
print("================")
print("Assignment 6.2")
print("================")
prior_comedy = 2 / 5
prior_action = 3 / 5

action_sentances = ["fast", "furious", "shoot", "furious", "shoot", "shoot", "fun", "fly", "fast", "shoot", "love"]
comdey_sentances = ["fun", "couple", "love", "love", "couple", "fly", "fast", "fun", "fun"]

# Calculating unigrams!
action_unigram = Counter(action_sentances)
comedy_unigram = Counter(comdey_sentances)
total_unigram = Counter(action_sentances + comdey_sentances)

document = ["fast", "couple", "shoot", "fly"]

# Initialise prob with posterior
action_posterior = np.log(prior_action)
comedy_posterior = np.log(prior_comedy)
action_count = sum(action_unigram.values())
comedy_count = sum(comedy_unigram.values())
# Loop over the document!
for word in document:
    action_posterior += np.log((action_unigram[word] + 1) / (action_count + len(total_unigram)))
    comedy_posterior += np.log((comedy_unigram[word] + 1) / (comedy_count + len(total_unigram)))


# Compare!
print("=============")
print("We determine the sentance to be: ")
if action_posterior > comedy_posterior:
    print("Action!")
else:
    print("Comedy!")


# ================
# Assignment 6.3
# ================

print("================")
print("Assignment 6.3")
print("================")

sentance = "A good good plot and great characters but poor acting".split()
classes = ["pos", "neg"]
documents = [
    ("pos",["good", "good", "good", "great", "great", "great"]),
    ("pos",["poor", "great", "great"]),
    ("neg",["good", "poor", "poor", "poor"]),
    ("neg",["good", "poor", "poor", "poor", "poor", "poor", "great", "great"]),
    ("neg",["poor", "poor"])
]

# Calculate counts in the documents
counters = {}
counters_bin = {}
counters["pos"] = Counter()
counters["neg"] = Counter()
counters_bin["pos"] = Counter()
counters_bin["neg"] = Counter()
for C in classes:
    for p, document in documents:
        if C == p:
            counters[C] += Counter(document)
            counters_bin[C] += Counter(set(document))

counters["tot"] = counters["neg"] + counters["pos"]
counters_bin["tot"] = counters_bin["neg"] + counters_bin["pos"]

# Calculate the posteriors
pos_prior = np.log2(2/5)
pos_bin_prior = np.log2(2/5)
neg_prior = np.log2(3/5)
neg_bin_prior = np.log2(3/5)
pos_count = sum(counters["pos"].values())
neg_count = sum(counters["neg"].values())
pos_bin_count = sum(counters_bin["pos"].values())
neg_bin_count = sum(counters_bin["neg"].values())

for word in sentance:
    # Make sure word is known!
    if word in counters["tot"].keys():
        pos_prior += np.log2( (counters["pos"][word] + 1) / (pos_count + len(counters["tot"])))
        neg_prior += np.log2( (counters["neg"][word] + 1) / (neg_count + len(counters["tot"])))

        pos_bin_prior += np.log2( (counters_bin["pos"][word] + 1) / (pos_bin_count + len(counters_bin["tot"])))
        neg_bin_prior += np.log2( (counters_bin["neg"][word] + 1) / (neg_bin_count + len(counters_bin["tot"])))


print("=============")
print("Normal Byes ")
print("pos: ", pos_prior, " neg: ", neg_prior)

if pos_prior > neg_prior:
    print("Positive!")
else:
    print("Negative!")
print("Binary Byes ")
print("pos: ",pos_bin_prior, " neg: ", neg_bin_prior)

if pos_bin_prior > neg_bin_prior:
    print("Positive!")
else:
    print("Negative!")
print("=============")

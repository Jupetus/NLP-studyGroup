import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

import torch
from torch.autograd import Variable
from torch import optim


def build_model(input_dim, output_dim):
    # We don't need the softmax layer here since CrossEntropyLoss already
    # uses it internally.
    model = torch.nn.Sequential()
    model.add_module("linear", torch.nn.Linear(input_dim, output_dim, bias=False))
    model.add_module("output", torch.nn.Sigmoid())
    return model

def train(model, loss, optimizer, x_val, y_val):
    x = Variable(x_val, requires_grad=False)
    y = Variable(y_val, requires_grad=False)

    # Reset gradient
    optimizer.zero_grad()

    # Forward
    fx = model.forward(x)
    output = loss.forward(fx, y)

    # Backward
    output.backward()

    # Update parameters
    optimizer.step()

    return output.data[0]


def predict(model, x_val):
    x = Variable(x_val, requires_grad=False)
    output = model.forward(x)
    return output.data.numpy().argmax(axis=1)

def stemMyStrings(words):
    stemmer = PorterStemmer()  # SnowballStemmer('english')
    newListSAS = []
    for word in words:
        newListSAS.append(stemmer.stem(word))
    return "".join(newListSAS)

def load_data():
    rTokenizer = RegexpTokenizer('\w+')
    path = "Dataset2.txt"
    Reviews = []
    with open(path) as f:
        for line in f.readlines():
            review = line.split('$')
            tokenized = rTokenizer.tokenize(review[2].lower().strip())
            stemmed = stemMyStrings(" ".join(tokenized))

            review[2] = " ".join(tokenized)
            Reviews.append((stemmed, review[1]))
    return Reviews

# Makes a sentance into a vector
def make_bow_vector(sentence, word_to_ix):
    vec = np.zeros(len(word_to_ix))
    for word in sentence:
        vec[word_to_ix[word]] += 1
    return vec

# target to a id?
def make_target(label):
    label_to_ix = {"neg": 0, "pos": 1}
    return label_to_ix[label.strip()]

def main():
    torch.manual_seed(42)

    all_data = load_data()

    offset = int(0.8 * len(all_data))
    train_data = [(all_data[tmp][0].split(), all_data[tmp][1]) for tmp in range(0, offset)]
    test_data =  [(all_data[tmp][0].split(), all_data[tmp][1]) for tmp in range(offset + 1, len(all_data))]

    word_to_ix = {}
    for sent, _ in train_data + test_data:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    n_features = len(word_to_ix)
    n_classes = 2

    trX = np.array([make_bow_vector(data[0], word_to_ix) for data in train_data])
    teX = np.array([make_bow_vector(data[0], word_to_ix) for data in test_data])
    trY = np.array([make_target(data[1]) for data in train_data])
    teY = np.array([make_target(data[1]) for data in test_data])
    trX = torch.from_numpy(trX).float()
    teX = torch.from_numpy(teX).float()
    trY = torch.from_numpy(trY).long()

    model = build_model(n_features, n_classes)
    loss = torch.nn.CrossEntropyLoss(size_average=True)
    optimizer = optim.SGD(model.parameters(), lr=0.025, momentum=0.9)
    batch_size = 100
    num_batches = len(train_data) // batch_size
    for epoch in range(100):
        cost = 0.
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            cost += train(model, loss, optimizer, trX[start:end], trY[start:end])

        predY = predict(model, teX)
        print("Epoch %d, cost = %f, acc = %.2f%%" % (epoch + 1, cost / 10, 100. * np.mean(predY == teY)))


if __name__ == "__main__":
    main()
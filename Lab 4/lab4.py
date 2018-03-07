import sys
import re
import string
import nltk
import os
import random
import matplotlib.pyplot as plt
from collections import Counter

# Random seed to ensure results stay the same.
random.seed(44339991231)

# Function to get the total number of words in all reviews.
def sum_words(counter):
    sum_words = 0
    for word in counter:
        sum_words += counter[word]
    return sum_words

# Function which goes through both the pos and neg folder and stores all of the words in a Counter().
def extract_corpus():
    path = folder +'/txt_sentoken/'
    print("Starting extraction")

    d_train = []
    d_test = []

    for foldername in os.listdir(path):

        # Ignore the hidden file .DS_Store
        if foldername != '.DS_Store':

            # Read in training data
            for filename in os.listdir(path + foldername + "/train"):

                with open((path + foldername + "/train/" + filename), "r") as f:

                    sentence_concat = []

                    for line in f:

                        # strip the first sentence of any punctuation, and split according to lines.
                        normalized_line = re.sub("[^\w']", " ", line.strip()).split()

                        # add the normalized sentence to the list for concatonated sentences
                        sentence_concat.extend(normalized_line)

                        # Add bigrams to the document vector as feature for improved accuracy
                        sentence_concat.extend(nltk.bigrams(normalized_line, pad_left=True, pad_right=True))

                    # Add the length of the document as feature for improved accuracy
                    d_counter = Counter(sentence_concat)

                    # As sentence_concat contains both words and bigrams, exclude the bigrams from the word count.
                    d_counter["$Length"] = sum([d_counter[term] for term in d_counter if (len(term) == 1)])
                    d_train.append((Counter(d_counter), foldername))

            # Read in testing data
            for filename in os.listdir(path + foldername + "/test"):

                with open((path + foldername + "/test/" + filename), "r") as f:

                    sentence_concat = []

                    for line in f:

                        # strip the first sentence of any punctuation, and split according to lines.
                        normalized_line = re.sub("[^\w']", " ", line.strip()).split()

                        # add the normalized sentence to the list for concatonated sentences
                        sentence_concat.extend(normalized_line)
                        sentence_concat.extend(nltk.bigrams(normalized_line, pad_left=True, pad_right=True))

                    d_counter = Counter(sentence_concat)
                    d_counter["$Length"] = sum([d_counter[term] for term in d_counter if (len(term) == 1)])
                    d_test.append((d_counter, foldername))

    print("Finished extraction")
    return d_train, d_test

def train(d_train):
    print("Starting training")
    weights = Counter()
    accuracies = []
    highest_weight = []

    # Counters to facilitate taking the average.
    total_weights = Counter()
    num_weight_updates = Counter()

    #Multiple passes
    for i in range(10):

        # Shuffling of training data
        random.shuffle(d_train)

        for train_d in d_train:
            score = 0.0

            for word, counts in train_d[0].items():
                score += counts * weights[word]

            # Set of if-else statements to decide if the document was correctly classified
            if score >= 0.0:
                sentiment = "pos"
            else:
                sentiment = "neg"

            if train_d[1] == sentiment:
                success = True
            else:
                success = False

            # If wrongly classified, update weights
            if success == False:
                if train_d[1] == "pos":
                    for word in train_d[0]:
                        weights[word] += (train_d[0][word])

                        # Update sum of weights & count of weight updates for averaging
                        total_weights[word] += weights[word]
                        num_weight_updates[word] += 1

                else:
                    for word in train_d[0]:
                        weights[word] -= (train_d[0][word])

                        # Update sum of weights & count of weight updates for averaging
                        total_weights[word] += weights[word]
                        num_weight_updates[word] += 1

        # Get the accuracy results for each iteration in order to plot the change of accuracy across iterations.
        result = test(weights, d_test, total_weights, num_weight_updates)
        accuracy = eval(result)
        accuracies.append(accuracy)

        highest_weight.append(weights.most_common(1))

        print("Finished iteration number " + str(i + 1))

    print("Finished training")
    return weights, total_weights, num_weight_updates, accuracies, highest_weight

def test(weights, d_test, total_weights, num_weight_updates):
    print("Starting testing")
    results = []

    # Testing is essentially training without updating the weights and, instead, passing on
    # whether the classifier was successful.
    for test_d in d_test:
        score = 0.0
        for word, counts in test_d[0].items():
            if num_weight_updates[word] != 0:
                score += counts * (total_weights[word]/num_weight_updates[word])
            else:
                score += 0.0
        if score >= 0.0:
            sentiment = "pos"
        else:
            sentiment = "neg"

        if test_d[1] == sentiment:
            success = True
        else:
            success = False

        results.append(success)

    print("Finished testing")
    return results

# Function to calculate the accuracy of the perceptron
def eval(results):
    results_counter = Counter(results)
    accuracy = results_counter[True]/(results_counter[True]+results_counter[False])
    return (accuracy*100)

# Function to plot the accuracy changes across iterations
def plot(accuracies):
    plt.plot(range(1, 11), accuracies)
    plt.xlabel('Iteration No.')
    plt.ylabel('Accuracy')
    plt.xticks(range(1, 11))
    plt.show()

# Take the "review_polarity" for the console input.
folder = sys.argv[1]

# D train & D test= [(X, Y)], where X is BoW for Document and Y is sentiment
d_train, d_test = extract_corpus()

# Results from training
weights, total_weights, num_weight_updates, accuracies, highest_weight = train(d_train)

# Results from testing
results = test(weights, d_test, total_weights, num_weight_updates)

# Calculate the accuracy
accuracy = eval(results)

print("Final accuracy: " + str(accuracy) + "%")
print("Highest weights: " + str(highest_weight))

# Plot the graph
plot(accuracies)

import numpy as np
import sys
import re
import string
import nltk
import random
from itertools import product
from sklearn.metrics import f1_score
from collections import Counter

# Random seed to ensure results stay the same.
random.seed(44339991231)

def word_tag_features(sentence):
    word_tags = Counter()
    for word_tag in sentence:
        word_tags[word_tag[0]+"_"+word_tag[1]] += 1
    return word_tags

def tag_tag_features(sentence):
    tag_tags = Counter()
    for i, word_tag in enumerate(sentence):
        if i>0:
            tag_tags[sentence[i-1][1] + "_" + sentence[i][1]] += 1
        else:
            tag_tags[str(None)+ "_" + sentence[i][1]] += 1
    return tag_tags

# Function to calculate the accuracy of the perceptron
def eval(results):
    results_counter = Counter(results)
    accuracy = results_counter[True]/(results_counter[True]+results_counter[False])
    return (accuracy*100)

# function to give me the total word count in each counter.
def sum_words(counter):
    sum_words = 0
    for word in counter:
        sum_words += counter[word]
    return sum_words

def extract_corpus(corpus_train, corpus_test):

    # initialize lists for the concatonated sentences and bigrams to be stored in
    sentence_concat_train = []
    sentence_concat_test = []
    bigrams = []

    with open(corpus_train, "r") as f:
        for line in f:

            # strip the first sentence of any punctuation, and split according to lines.
            normalized_line = re.sub('\t', '\t', line.strip()).split('\t', 2)

            i = 0
            zippedList = []

            for elem in normalized_line:

                elem = re.sub("[^\w]", " ",  elem).split()

                if i == 0:
                    zippedList.append(elem)
                else:
                    zippedList.append(elem)

                i+=1

            zippedList = list(zip(zippedList[0], zippedList[1]))

            # add the normalized sentence to the list for concatonated sentences
            sentence_concat_train.append(zippedList)

            # use nltk to form bigrams for all word pairs in each sentence of the corpus.
            bigrams.append(nltk.bigrams(zippedList, pad_left=True, pad_right=True))
        sentence_concat_train[:] = [item for item in sentence_concat_train if len(item) != 0]

    # Read in testing data
    with open(corpus_train, "r") as f:
        for line in f:

            # strip the first sentence of any punctuation, and split according to lines.
            normalized_line = re.sub('\t', '\t', line.strip()).split('\t', 2)

            i = 0
            zippedList = []

            for elem in normalized_line:

                elem = re.sub("[^\w]", " ",  elem).split()

                if i == 0:
                    zippedList.append(elem)
                else:
                    zippedList.append(elem)

                i+=1

            zippedList = list(zip(zippedList[0], zippedList[1]))

            # add the normalized sentence to the list for concatonated sentences
            sentence_concat_test.append(zippedList)

            # # use nltk to form bigrams for all word pairs in each sentence of the corpus.
            #     bigrams.append(nltk.bigrams(zippedList, pad_left=True, pad_right=True))
        sentence_concat_test[:] = [item for item in sentence_concat_test if len(item) != 0]

    return sentence_concat_train, sentence_concat_test

def train(d_train):
    print("Starting training")

    possible_tags = ["O", "LOC", "PER", "MISC", "ORG"]

    weights = Counter()
    accuracies = []
    highest_weight = []
    lowest_weight = []

    # Counters to facilitate taking the average.
    total_weights = Counter()
    num_weight_updates = Counter()

    #Multiple passes
    for i in range(10):

        # Shuffling of training data
        random.shuffle(d_train)

        # For each labelled line in the training set
        for train_d in d_train:

            # Use the word_tag_features function to get the Counter() for the correct word_labels & label_labels
            corr_word_tag_count = word_tag_features(train_d)
            corr_tag_tag_count = tag_tag_features(train_d)

            length = len(train_d)

            # The words found in the line
            words = [x[0] for x in train_d]

            # The relevant correct tags found in the line
            correct_tags = [x[1] for x in train_d]

            # All possible combinations of "O", "LOC", "PER", "MISC", "ORG", shuffled.
            tag_combinations = list(product(possible_tags, repeat = length))
            random.shuffle(tag_combinations)

            tags = []

            # Go through every single possible label combination, calculate the dot product score
            # for that combination, and then select the highest scoring combination as the prediction.
            for combo in tag_combinations:

                score = 0.0

                # Counter to keep track of the scores for each of the possible combinations.
                word_label_count = Counter()

                j = 0

                #tag-tag count
                for word_tag in combo:
                    if j>0:
                        word_label_count[combo[j-1]+ "_" + combo[j]] += 1
                        score = word_label_count[combo[j-1] + "_" + combo[j]] * weights[combo[j-1] + "_" + combo[j]]

                    else:
                        word_label_count[str(None)+ "_" + combo[j]] += 1
                        score = word_label_count[str(None)+ "_" + combo[j]] * weights[str(None)+ "_" + combo[j]]

                    j += 1

                j = 0
                #word-tag count
                for word in words:
                    word_label_count[word + "_" + combo[j]] += 1
                    score += word_label_count[word + "_" + combo[j]] * weights[word + "_" + combo[j]]
                    word_label_count[word + "_" + combo[j]] = score
                    j += 1

                tags.append((word_label_count, sum_words(word_label_count)))

            # Get the argmax of the word_label_count Counter, which is the predicted label sequence
            prediction = tag_combinations[np.argmax([poss[1] for poss in tags])]

            feat_predicted = Counter()

            # Re-compute the Counter for the predicted label sequence
            j = 0
            for word in words:
                feat_predicted[word + "_" + prediction[j]] = feat_predicted[word + "_" + prediction[j]] * weights[word + "_" + prediction[j]]
                j += 1


            # The correct feature vector
            feat_diff = corr_word_tag_count + corr_tag_tag_count

            # Are they the same? If not, add the difference to the weights.
            feat_diff.subtract(feat_predicted)

            # if-else statement to decide if the labels were correctly identified
            # if the label sequence was correctly predicted, then the resulting Counter should be 0.
            if sum_words(feat_diff) == 0:
                success = True
            else:
                success = False

            # If wrongly classified, update weights
            if success == False:
                for word_tag in feat_diff:
                    weights[word_tag] += feat_diff[word_tag]

                    # Update sum of weights & count of weight updates for averaging
                    total_weights[word_tag] += weights[word_tag]
                    num_weight_updates[word_tag] += 1

        print("Finished iteration number " + str(i + 1))

    # Averaging over multiple passes for the weights.
    for word_tag in weights:
        weights[word_tag] = total_weights[word_tag]/num_weight_updates[word_tag]


    # Get the accuracy results for each iteration in order to plot the change of accuracy across iterations.
    correct_labels, predicted_labels = test(weights, sentence_concat_test)
    highest_weight.append(weights.most_common(10))
    lowest_weight.append(weights.most_common()[-10:-1])
    print("Finished training")
    return weights, total_weights, num_weight_updates, correct_labels, predicted_labels, highest_weight, lowest_weight

def test(weights, d_test):
    print("Starting testing")
    correct_labels = []
    predicted_labels = []
    corr_word_tag_count = Counter()
    corr_tag_tag_count = Counter()

    possible_tags = ["O", "LOC", "PER", "MISC", "ORG"]

    # Testing is essentially training without updating the weights and, instead, passing on
    # whether the classifier was successful.
    for test_d in d_test:

        corr_word_tag_count += word_tag_features(test_d)
        corr_tag_tag_count += tag_tag_features(test_d)

        length = len(test_d)
        words = [x[0] for x in test_d]
        correct_tags = [x[1] for x in test_d]

        tag_combinations = list(product(possible_tags, repeat = length))
        random.shuffle(tag_combinations)

        tags = []

        for combo in tag_combinations:
            word_label_count = Counter()
            j = 0
            for word_tag in combo:
                if j>0:
                    word_label_count[combo[j-1]+ "_" + combo[j]] += 1
                    score = word_label_count[combo[j-1] + "_" + combo[j]] * weights[combo[j-1] + "_" + combo[j]]

                else:
                    word_label_count[str(None)+ "_" + combo[j]] += 1
                    score = word_label_count[str(None)+ "_" + combo[j]] * weights[str(None)+ "_" + combo[j]]

                j += 1

            j = 0
            for word in words:
                word_label_count[word + "_" + combo[j]] += 1
                score = word_label_count[word + "_" + combo[j]] * weights[word + "_" + combo[j]]
                word_label_count[word + "_" + combo[j]] = score
                j += 1
            tags.append((word_label_count, sum_words(word_label_count)))

        prediction = tag_combinations[np.argmax([poss[1] for poss in tags])]

        feat_predicted = Counter()
        j = 0
        for word in words:
            feat_predicted[word + "_" + prediction[j]] = feat_predicted[word + "_" + prediction[j]] * weights[word + "_" + prediction[j]]
            j += 1

        feat_diff = corr_word_tag_count + corr_tag_tag_count
        feat_diff.subtract(feat_predicted)

        correct_labels.append(correct_tags)
        predicted_labels.append(list(prediction))

    correct_labels = [x for sublist in correct_labels for x in sublist]
    predicted_labels = [x for sublist in predicted_labels for x in sublist]

    print("Finished testing")
    return correct_labels, predicted_labels

corpus_train = sys.argv[1]
corpus_test = sys.argv[2]
sentence_concat_train, sentence_concat_test = extract_corpus(corpus_train, corpus_test)
weights, total_weights, num_weight_updates, correct_labels, predicted_labels, highest_weight, lowest_weight = train(sentence_concat_train)
f1_metric = f1_score(correct_labels, predicted_labels, average='micro', labels=["O", "LOC", "PER", "MISC", "ORG"])

print(" ------------------------ ")
print("Highest weighted features:")
print(highest_weight)
print(" ------------------------ ")
print("F1_Score:")
print(f1_metric)
print(" ------------------------ ")

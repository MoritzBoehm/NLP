import numpy
import sys
import re
import string
import nltk
import os
from collections import Counter

# Function to get the total number of words in all reviews.
def sum_words(counter):
    sum_words = 0
    for word in counter:
        sum_words += counter[word]
    return sum_words

# Function which goes through both the pos and neg folder and stores all of the words in a Counter().
def extract_corpus():
    path = 'review_polarity/txt_sentoken/'

    sentence_concat = []
    sentiment_inc = []

    for foldername in os.listdir(path):

        # Ignore the hidden file .DS_Store
        if foldername != '.DS_Store':

            for filename in os.listdir(path + foldername):

                with open((path + foldername + "/" + filename), "r") as f:
                    for line in f:

                        # strip the first sentence of any punctuation, and split according to lines.
                        normalized_line = re.sub("[^\w']", " ", line.strip()).split()

                        # add the normalized sentence to the list for concatonated sentences
                        sentence_concat.extend(normalized_line)

            for elem in sentence_concat:
                sentiment_inc.append((elem, foldername))

    return sentiment_inc


extracted_corpus = extract_corpus()
only_words = [i[0] for i in extracted_corpus[:]]
unigram = Counter(only_words)

Pos_Weights = []
Neg_Weights = []

for word_with_sentiment in extracted_corpus:
    if word_with_sentiment[1] == 'pos':
        Pos_Weights.append((word_with_sentiment[0], 0))
    else:
         Neg_Weights.append((word_with_sentiment[0], 0))

print("No. of unique words: " + str(len(unigram)))
print("No. of words in Pos Weight Vector: " + str(len(Pos_Weights)))
print("No. of words in Neg Weight Vector: " + str(len(Neg_Weights)))
print("No. of words in Counter(): " + str(sum_words(unigram)))
print("Pos_Weights + Neg_Weights: " + str(len(Neg_Weights) + len(Pos_Weights)))
print("Do weight vectors match? " + str(sum_words(unigram) == (len(Neg_Weights) + len(Pos_Weights))))

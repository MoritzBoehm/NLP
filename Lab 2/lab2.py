import numpy
import sys
import re
import string
import nltk
from collections import Counter

def extract_corpus(corpus):

    # initialize lists for the concatonated sentences and bigrams to be stored in
    sentence_concat = []
    bigrams = []


    with open(corpus, "r") as f:
        for line in f:

            # strip the first sentence of any punctuation, and split according to lines.
            normalized_line = re.sub('['+string.punctuation+']', '', line.strip()).split()

            # add the normalized sentence to the list for concatonated sentences
            sentence_concat.extend(normalized_line)

            # use nltk to form bigrams for all word pairs in each sentence of the corpus.
            bigrams.extend(nltk.bigrams(normalized_line, pad_left=True, pad_right=True))

    unigram = Counter(sentence_concat)
    bigram = Counter(bigrams)
    return unigram, bigram

def extract_questions(questions):
    with open(questions, "r") as f:

        # similar to how the corpus is extracted, except I use a special regex to leave underscores and backslashes in.
        return [re.sub("[^a-zA-Z0-9_\s/]", '', line.strip()).split() for line in f]

# function to give me the total word count in each counter.
def sum_words(counter):
    sum_words = 0
    for word in counter:
        sum_words += counter[word]
    return sum_words

# function to calculate the unigram language model probabilities.
def unigram_probability(questions, counter):
    sum_of_words = sum_words(counter)
    options = []
    best_words = []

    # add the last element of each question sentence (the word choices) into a separate list.
    for sentence in questions:
        options.append(sentence[-1])

    # get each candidate word from the options list.
    for option in options:
        option = option.split('/')
        word_1 = option[0]
        word_2 = option[1]

        # calculate unigram probability, the no. of occurences of a word over the total number of words.
        word1_prob = counter[word_1]/sum_of_words
        word2_prob = counter[word_2]/sum_of_words

        if word1_prob > word2_prob:
            best_words.append(word_1)
        else:
            best_words.append(word_2)

    return best_words

# function to calculate the bigram language model probabilities.
def bigram_probability(questions, unigrams, bigrams, smoothing):
    unique_words = len(unigrams.keys())
    options = []
    question_bigrams = []
    best_words = []

    if smoothing:
        smoothing = 1.0
    else:
        smoothing = 0.0

    for sentence in questions:

        # Index of where the word needs to go in the sentence.
        blank_pos = sentence.index("____")

        # same procedure as for the unigram model.
        options = sentence[-1]
        option = options.split('/')
        word_1 = option[0]
        word_2 = option[1]

        # get the words to the left and right of the blank, and zip them together, and store in a list now containing
        # both bigrams for each word.
        bigram_word_1 = [sentence[(blank_pos-1)], word_1, sentence[(blank_pos+1)]]
        bigram_word_1 = list(zip(bigram_word_1[:-1], bigram_word_1[1:]))

        bigram_word_2 = [sentence[(blank_pos-1)], word_2, sentence[(blank_pos+1)]]
        bigram_word_2 = list(zip(bigram_word_2[:-1], bigram_word_2[1:]))

        # Calculate the bigram probabilities as per the formula.
        bigram1_probability1 = (bigrams[bigram_word_1[0]]+smoothing)/(unigrams[bigram_word_1[0][0]]+smoothing*unique_words)
        bigram1_probability2 = (bigrams[bigram_word_1[1]]+smoothing)/(unigrams[bigram_word_1[1][0]]+smoothing*unique_words)
        bigram1_probability = bigram1_probability1*bigram1_probability2

        bigram2_probability1 = (bigrams[bigram_word_2[0]]+smoothing)/(unigrams[bigram_word_2[0][0]]+smoothing*unique_words)
        bigram2_probability2 = (bigrams[bigram_word_2[1]]+smoothing)/(unigrams[bigram_word_2[1][0]]+smoothing*unique_words)
        bigram2_probability = bigram2_probability1*bigram2_probability2

        # check to make sure that at least one probability is > 0, and to ensure they do not have the same probabilities.
        if (bigram1_probability != bigram2_probability):
            if bigram1_probability > bigram2_probability:
                best_words.append(word_1)
            else:
                best_words.append(word_2)
        # Check for a half-correct answer, as this statement is only reached if the two probabilities are not equal,
        # but also non-zero.
        elif (bigram1_probability != 0.0):
            best_words.append("Halfway-Tie")
        else:
            best_words.append("No correct answer")

    return best_words



corpus = sys.argv[1]
questions = sys.argv[2]

unigrams, bigrams = extract_corpus(corpus)
questions_extracted = extract_questions(questions)


# Console output
print("-------- UNIGRAM LANGUAGE MODEL --------")
print(unigram_probability(questions_extracted, unigrams))

print("-------- BIGRAM LANGUANGE MODEL --------")
print(bigram_probability(questions_extracted, unigrams, bigrams, False))

print("-------- BIGRAM LANGUANGE MODEL ADD-1 --------")
print(bigram_probability(questions_extracted, unigrams, bigrams, True))

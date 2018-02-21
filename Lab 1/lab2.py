import numpy
import sys
import re
import string
import nltk
from collections import Counter

def debug_corpus_extraction(corpus_extracted):
    print(corpus_extracted[0])
    print("Length: " + str(len(corpus_extracted)))


def debug_unigram(unigram_prob):
    print(unigram_prob.most_common(3))
    print("Length: " + str(len(unigram_prob)))
    print("Sum: " + str(sum_words(unigram_prob)))
    print("No. of words: " + str(len(unigram_prob)))

def extract_corpus(corpus):
    sentence_concat = []
    bigrams = []
    with open(corpus, "r") as f:
        for line in f:
            normalized_line = re.sub('['+string.punctuation+']', '', line.strip()).split()
            sentence_concat.extend(normalized_line)
            bigrams.extend(nltk.bigrams(normalized_line, pad_left=True, pad_right=True))
    unigram = Counter(sentence_concat)
    bigram = Counter(bigrams)
    return unigram, bigram

def extract_questions(questions):
    with open(questions, "r") as f:
        return [re.sub("[^a-zA-Z0-9_\s/]", '', line.strip()).split() for line in f]

def sum_words(counter):
    sum_words = 0
    for word in counter:
        sum_words += counter[word]
    return sum_words

def unigram_probability(questions, counter):
    sum_of_words = sum_words(counter)
    options = []
    best_words = []

    for sentence in questions:
        options.append(sentence[-1])

    for option in options:
        option = option.split('/')
        word_1 = option[0]
        word_2 = option[1]

        word1_prob = counter[word_1]/sum_of_words
        word2_prob = counter[word_2]/sum_of_words

        if word1_prob > word2_prob:
            best_words.append(word_1)
        else:
            best_words.append(word_2)

    return best_words

def bigram_probability(questions, unigrams, bigrams, smoothing):
    unique_words = len(unigrams.keys())
    options = []
    question_bigrams = []
    best_words = []

    if smoothing:
        smoothing = 1.0
    else:
        smoothing = 0.0

    print()

    for sentence in questions:
        blank_pos = sentence.index("____")
        options = sentence[-1]
        option = options.split('/')
        word_1 = option[0]
        word_2 = option[1]

        bigram_word_1 = [sentence[(blank_pos-1)], word_1, sentence[(blank_pos+1)]]
        bigram_word_1 = list(zip(bigram_word_1[:-1], bigram_word_1[1:]))
        bigram_word_2 = [sentence[(blank_pos-1)], word_2, sentence[(blank_pos+1)]]
        bigram_word_2 = list(zip(bigram_word_2[:-1], bigram_word_2[1:]))

        bigram1_probability1 = (bigrams[bigram_word_1[0]]+smoothing)/(unigrams[bigram_word_1[0][0]]+smoothing*unique_words)
        bigram1_probability2 = (bigrams[bigram_word_1[1]]+smoothing)/(unigrams[bigram_word_1[1][0]]+smoothing*unique_words)
        bigram1_probability = bigram1_probability1*bigram1_probability2

        bigram2_probability1 = (bigrams[bigram_word_2[0]]+smoothing)/(unigrams[bigram_word_2[0][0]]+smoothing*unique_words)
        bigram2_probability2 = (bigrams[bigram_word_2[1]]+smoothing)/(unigrams[bigram_word_2[1][0]]+smoothing*unique_words)
        bigram2_probability = bigram2_probability1*bigram2_probability2

        if (bigram1_probability and bigram2_probability != 0.0) and (bigram1_probability != bigram2_probability):
            if bigram1_probability > bigram2_probability:
                best_words.append(word_1)
            else:
                best_words.append(word_2)
        else:
            best_words.append("No correct answer")

    return best_words



corpus = sys.argv[1]
questions = sys.argv[2]

unigrams, bigrams = extract_corpus(corpus)
questions_extracted = extract_questions(questions)


print("-------- UNIGRAM LANGUAGE MODEL --------")
print(unigram_probability(questions_extracted, unigrams))

print("-------- BIGRAM LANGUANGE MODEL --------")
print(bigram_probability(questions_extracted, unigrams, bigrams, False))

print("-------- BIGRAM LANGUANGE MODEL ADD-1 --------")
print(bigram_probability(questions_extracted, unigrams, bigrams, True))

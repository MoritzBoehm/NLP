from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from utils.pos_counts import complex_words_PoS_count
import numpy as np
import spacy
import re
import string


class Model(object):

    def __init__(self, language):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
        else:  # spanish
            self.avg_word_length = 6.2

        self.tag_counts = Counter()

        # Implemented a random forest classifier due to it's success in the CWI Shared Task 2016.
        self.model = RandomForestClassifier()


    def extract_features(self, sentence, model):

        word = sentence['target_word']
        tokenized_word = word.split()
        whole_sentence = sentence['sentence']
        if int(sentence['start_offset']) > 3:
            partial_sentence = whole_sentence[:(int(sentence['start_offset']))]
            if partial_sentence:
                bigram = (partial_sentence.split()[-1], word)

        else:
            bigram = ("None", word)


        len_chars = len(word) / self.avg_word_length
        len_tokens = len(word.split(' '))

########################################################################################################

        # Get the word embedding for the target word to append to the feature vector
        # word embeddings are word-level, but could be expanded to sentence level. If the word is
        # a phrase, then it will be a sum of the word embeddings
        # Sometimes you get one word
        if bigram[0] == "None":
            if len(tokenized_word) == 1:
                tokenized_word = word
                if word in model:
                    word_embedding = (np.zeros(300), model[word])
                else:
                    word_embedding = (np.zeros(300), np.zeros(300))
            # And sometimes you get more than one word
            elif (len(tokenized_word) > 1):
                dotProduct = []
                for elem in tokenized_word:
                    if elem in model:
                        dotProduct.append(model[elem])
                    else:
                        dotProduct.append(np.zeros(300))

                var_name = np.zeros(300)
                for i in range(len(dotProduct)):
                    var_name = np.add(var_name, dotProduct[i])

                word_embedding = (np.zeros(300), var_name)
        else:
            if len(tokenized_word) == 1:
                tokenized_word = word
                if word in model:
                    if bigram[0] in model:
                        word_embedding = (model[bigram[0]], model[word])
                    else:
                        word_embedding = (np.zeros(300), model[word])
                else:
                    if bigram[0] in model:
                        word_embedding = (model[bigram[0]], np.zeros(300))
                    else:
                        word_embedding = (np.zeros(300), np.zeros(300))
            # And sometimes you get more than one word
            elif (len(tokenized_word) > 1):
                dotProduct = []
                for elem in tokenized_word:
                    if elem in model:
                        dotProduct.append(model[elem])
                    else:
                        dotProduct.append(np.zeros(300))

                var_name = np.zeros(300)
                for i in range(len(dotProduct)):
                    var_name = np.add(var_name, dotProduct[i])

                if bigram[0] in model:
                    word_embedding = (model[bigram[0]], var_name)
                else:
                    word_embedding = (np.zeros(300), var_name)



########################################################################################################






########################################################################################################

        # Append the features to the feature vector
        features = [len_chars, len_tokens]
        word_embedding = np.concatenate((word_embedding[0], word_embedding[1]), axis = 0)
        for elem in word_embedding:
            features.append(elem)

        return features

    def train(self, trainset, model):
        X = []
        Y = []
        # self.tag_count = complex_words_PoS_count(trainset)
        # print(self.tag_count)
        for sent in trainset:
            X.append(self.extract_features(sent, model))
            Y.append(sent['gold_label'])

        self.model.fit(X, Y)

    def test(self, testset,model):
        X = []
        for sent in testset:
            X.append(self.extract_features(sent, model))

        return self.model.predict(X)

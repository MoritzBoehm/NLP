from sklearn.ensemble import RandomForestClassifier
import numpy as np

class Baseline(object):

    def __init__(self, language):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
        else:  # spanish
            self.avg_word_length = 6.2

        # Implemented a random forest classifier due to it's success in the CWI Shared Task 2016.
        self.model = RandomForestClassifier()

    def extract_features(self, word, model):
        len_chars = len(word) / self.avg_word_length
        len_tokens = len(word.split(' '))

########################################################################################################

        # Get the word embedding for the target word to append to the feature vector
        # word embeddings are word-level, but could be expanded to sentence level. If the word is
        # a phrase, then it will be a sum of the word embeddings
        tokenized_word = word.split()
        # Sometimes you get one word
        if len(tokenized_word) == 1:
            tokenized_word = word
            if word in model:
                word_embedding = model[word]
            else:
                word_embedding = np.zeros(300)
        # And sometimes you get more than one word
        elif (len(tokenized_word) > 1):
            dotProduct = []
            for elem in tokenized_word:
                if elem in model:
                    dotProduct.append(model[elem])
                else:
                    dotProduct.append(np.zeros(300))
            for i in range(len(dotProduct)):
                if i != (len(dotProduct) - 1):
                    word_embedding = np.dot((dotProduct[i]), (dotProduct[i+1]))
                elif i == (len(dotProduct) - 1):
                    word_embedding = np.dot(word_embedding, (dotProduct[i]))

########################################################################################################

        # Append the features to the feature vector
        features = [len_chars, len_tokens]
        for elem in word_embedding:
            features.append(elem)

        return features

    def train(self, trainset, model):
        X = []
        Y = []
        for sent in trainset:
            X.append(self.extract_features(sent['target_word'], model))
            Y.append(sent['gold_label'])

        self.model.fit(X, Y)

    def test(self, testset,model):
        X = []
        for sent in testset:
            X.append(self.extract_features(sent['target_word'], model))

        return self.model.predict(X)

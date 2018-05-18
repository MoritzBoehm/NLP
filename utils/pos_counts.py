from collections import Counter
import spacy


def complex_words_PoS_count(dataset):
        tag_count = Counter()
        for sent in dataset:
            word = sent['target_word']
            tokenized_word = word.split()
            sentence = sent['sentence']
            gold_label = sent['gold_label']
            nlp = spacy.load('en_core_web_sm')
            doc = nlp(sentence)

            tags = [(token.text, token.tag_) for token in doc]
            if gold_label == "1":
                if len(tokenized_word) == 1:
                    tokenized_word = word
                    for tag in tags:
                        if tag[0] == word:
                            tag_count[tag[1]] += 1
                # And sometimes you get more than one word
                elif (len(tokenized_word) > 1):
                    for word in tokenized_word:
                        for tag in tags:
                            if tag[0] == word:
                                tag_count[tag[1]] += 1

        return tag_count

import spacy

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load('en')

doc = nlp(u"Peach emoji is where it has always been. Peach is the superior "
          u"emoji. It's outranking eggplant 🍑 ")

## Essentially List Indexing
assert doc[0].text == u'Peach'
assert doc[1].text == u'emoji'
assert doc[-1].text == u'🍑'
assert doc[17:19].text == u'outranking eggplant'
assert list(doc.noun_chunks)[0].text == u'Peach emoji'

sentences = list(doc.sents)
assert len(sentences) == 3
assert sentences[1].text == u'Peach is the superior emoji.'

# PoS tagging & flagging
doc = nlp(u'Apple is looking at buying U.K. startup for $1 billion')
apple = doc[0]

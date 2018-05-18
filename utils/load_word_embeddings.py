import numpy as np

def load_word_embeddings(language):
    print("Loading word embeddings")

    if language == 'english':
        filePath =  "datasets/english/glove.6B.300d.txt"
    elif language == 'spanish':
        filePath = "datasets/spanish/glove-sbwc.i25.vec"

    f = open(filePath,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding

    print("Done. ",len(model)," words loaded!")
    return model

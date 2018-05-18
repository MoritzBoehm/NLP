from utils.dataset import Dataset
from utils.baseline import Baseline
from utils.scorer import report_score
from utils.model import Model
from utils.load_word_embeddings import load_word_embeddings


def execute_demo(language):
    if language == 'english':
        word_emb = load_word_embeddings('english')
    elif language == 'spanish':
        word_emb = load_word_embeddings('spanish')

    data = Dataset(language)

    print("{}: {} training - {} dev".format(language, len(data.trainset), len(data.devset)))

    #for sent in data.trainset:
        # Gold label -> 0 if the word is not complex, 1 if the word is complex.
        #print(sent['sentence'], sent['target_word'], sent['gold_label'])

    baseline = Baseline(language)

    model = Model(language)

    model.train(data.trainset, word_emb)

    predictions = model.test(data.devset, word_emb)

    gold_labels = [sent['gold_label'] for sent in data.devset]

    report_score(gold_labels, predictions)


if __name__ == '__main__':
    execute_demo('english')
    execute_demo('spanish')

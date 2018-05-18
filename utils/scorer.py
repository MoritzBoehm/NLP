import sklearn


def report_score(gold_labels, predicted_labels):
    print("Generating report")

    macro_F1 = round(sklearn.metrics.f1_score(gold_labels, predicted_labels, average='macro'), 3)
    balancedAccuracy, precision, recall = getAccuracy(gold_labels, predicted_labels)
    gscore, accuracy = getGScore(gold_labels, predicted_labels, recall)


    print("Accuracy: " + str(accuracy))
    print("Balanced accuracy: " + str(balancedAccuracy))
    print("macro-F1: {:.3f}".format(macro_F1))
    print("G Score: " + str(gscore))
    print("Precision: " + str(precision))
    print("{:^10}{:^10}{:^10}{:^10}{:^10}".format("Accuracy", "BalancedAccuracy", "F-Score", "G-Score", " Precision"))
    print('-' * 70)
    print("{:^15}{:^15}{:^15}{:^15}{:^15}".format(accuracy, balancedAccuracy, macro_F1, gscore, precision))
    print()

def getGScore(gold_labels, predicted_labels, recall):
    accuracy = sklearn.metrics.accuracy_score(gold_labels, predicted_labels)
    gscore = 2*((accuracy*recall)/(accuracy+recall))
    return [round(gscore, 3), round(accuracy, 3)]

def getAccuracy(gold_labels, predicted_labels):
    truePositive = 0
    trueNegative = 0
    falsePositive = 0
    falseNegative = 0

    length0 = len([label for label in gold_labels if label == '0'])
    length1 = len([label for label in gold_labels if label == '1'])

    for i in range(len(gold_labels)):
        gold_label = gold_labels[i]
        pred_label = predicted_labels[i]

        if pred_label == '0':
            if gold_label == '1':
                falseNegative += 1

            elif gold_label == '0':
                trueNegative += 1

        elif pred_label == '1':
            if gold_label == '1':
                truePositive += 1

            elif gold_label == '0':
                falsePositive += 1

    accuracy = round(((truePositive/length1) + (trueNegative/length0))/2, 3)
    precision = round((truePositive/(truePositive + falsePositive)), 3)
    recall = (truePositive/(truePositive+falseNegative))

    return [accuracy, precision, recall]

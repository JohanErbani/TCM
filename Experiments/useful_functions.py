import sys
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import torch
import numpy as np
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt


def print_list(l):
    w = '['
    for s in ["{:.2f}".format(x) for x in l]:
        w += s + ' '
    w = w[:-1]
    w += ']\n'
    print(w)


def plot_test(train_loss, test_loss, f1s_micro, f1s_macro, f1s_weighted, M):
    plt.figure()

    plt.subplot(2, 2, 1)
    plt.plot(train_loss)
    plt.title('train loss')

    plt.subplot(2, 2, 2)
    plt.plot(test_loss)
    plt.title('test loss')

    plt.subplot(2, 2, 3)

    plt.plot(f1s_micro, label='micro')
    plt.plot(f1s_macro, label='macro')
    plt.plot(f1s_weighted, label='weighted')
    plt.title('f1s')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.imshow(M)
    plt.axis('off')
    plt.title('TCM')

    plt.tight_layout()

    plt.show()


"""
###################################################################
METRICS ###########################################################
###################################################################
"""


def one_hot(labels, C):
    n = len(labels)
    one_hot_labels = torch.zeros((n, C))
    one_hot_labels[torch.arange(n), labels] = 1.
    return one_hot_labels


def hard(predictions, thresholds):
    n = len(predictions)
    binary_predictions = torch.zeros_like(predictions)
    binary_predictions[torch.arange(n), torch.argmax(predictions, dim=1)] = 1.
    for c, threshold in enumerate(thresholds):
        binary_predictions[threshold <= predictions[:, c], c] = 1.
    return binary_predictions


class collect_instances:
    def __init__(self, C):
        self.label_list = []
        self.prediction_list = []
        self.C = C

    def update(self, labels, predictions):
        self.label_list.append(labels.cpu())
        self.prediction_list.append(predictions.cpu())

    def get_labels(self):
        self.labels = torch.cat(self.label_list, dim=0)
        return self.labels

    def get_predictions(self):
        self.predictions = torch.cat(self.prediction_list, dim=0)
        return self.predictions

    def get_hard_predictions(self, thresholds=None):
        self.predictions = torch.cat(self.prediction_list, dim=0)
        if thresholds is None:
            thresholds = self.get_thresholds()
        binary_predictions = hard(self.predictions, thresholds)
        return binary_predictions

    def get_thresholds(self):
        C_thresholds = []
        self.labels = torch.cat(self.label_list, dim=0)
        self.predictions = torch.cat(self.prediction_list, dim=0)

        for i in range(self.C):
            fpr, tpr, thresholds = precision_recall_curve(self.labels[:, i].numpy(), self.predictions[:, i].numpy())
            gmeans = np.sqrt(tpr * (1 - fpr))
            idx = np.argmax(gmeans)
            #
            C_thresholds.append(thresholds[idx])

        return C_thresholds


def all_metrics(labels, binary_predictions):
    labels, binary_predictions = labels.tolist(), binary_predictions.tolist()

    f1 = f1_score(labels, binary_predictions, average=None)
    f1_micro = f1_score(labels, binary_predictions, average='micro')
    f1_macro = f1_score(labels, binary_predictions, average='macro')
    f1_weighted = f1_score(labels, binary_predictions, average='weighted')
    precision = precision_score(labels, binary_predictions, average=None)
    recall = recall_score(labels, binary_predictions, average=None)

    precision_micro = precision_score(labels, binary_predictions, average='micro')
    recall_micro = recall_score(labels, binary_predictions, average='micro')
    precision_macro = precision_score(labels, binary_predictions, average='macro')
    recall_macro = recall_score(labels, binary_predictions, average='macro')
    precision_weighted = precision_score(labels, binary_predictions, average='weighted')
    recall_weighted = recall_score(labels, binary_predictions, average='weighted')

    print('f1')
    print_list(f1)
    print('Recall')
    print_list(recall)
    print('Precision')
    print_list(precision)
    print('f1s')
    print_list([f1_micro, f1_macro, f1_weighted])
    print('Recalls')
    print_list([recall_micro, recall_macro, recall_weighted])
    print('Precisions')
    print_list([precision_micro, precision_macro, precision_weighted])

    return f1, f1_micro, f1_macro, f1_weighted


"""
###################################################################
DATA ##############################################################
###################################################################
"""


def list2vect(l, class_names):
    intersection = list(set(l) & set(class_names))
    if not intersection:
        return np.nan
    else:
        new_list = [0 for _ in range(len(class_names))]
        for idx, c in enumerate(class_names):
            if c in intersection:
                new_list[idx] = 1
        return new_list


def int2vect(x, C):
    new_list = [0 for _ in range(C)]
    new_list[x] = 1
    return new_list


def str2list(text):
    text = text.replace('[', '')
    text = text.replace(']', '')
    text = text.replace(' ', '')
    return text.split(',')


def EXIST(annotators):
    class_names = ['-', 'STEREOTYPING-DOMINANCE', 'IDEOLOGICAL-INEQUALITY', 'OBJECTIFICATION',
                   'MISOGYNY-NON-SEXUAL-VIOLENCE', 'SEXUAL-VIOLENCE']

    label = []

    for annotator in annotators:
        tmp = [0 for _ in range(len(class_names))]
        for idx, c in enumerate(class_names):
            if c in annotator:
                tmp[idx] = 1
        label += tmp
    if all(x == 0 for x in label):
        return -1
    else:
        return label


def get_data(df, X_name, y_name, class_names=None, maximal_C=None):
    df = df[[X_name, y_name]]
    is_int = isinstance(df[y_name][0], np.integer) or isinstance(df[y_name][0], int)

    if (not is_int) and isinstance(df[y_name][0], str):
        df.loc[:, y_name] = df[y_name].apply(str2list)

    if class_names is None:
        class_distribution = Counter(df[y_name].sum())
        class_names = []

        class_distribution_most_common = class_distribution.most_common(maximal_C)

        print('classes:', class_distribution_most_common)
        for i in class_distribution_most_common:
            class_names.append(i[0])

    if is_int:
        df.loc[:, y_name] = df[y_name].apply(lambda label: int2vect(label, len(class_names)))
    elif isinstance(df[y_name][0], list):
        df.loc[:, y_name] = df[y_name].apply(lambda label: list2vect(label, class_names))
    elif isinstance(df[y_name][0], str):
        df.loc[:, y_name] = df[y_name].apply(lambda label: list2vect([label], class_names))

    df = df.dropna().reset_index()

    C = len(class_names)

    return df, class_names, C


def save_res(train_loss, test_loss, f1s_micro, f1s_macro, f1s_weighted, experience_name):
    data = {
        'train_loss': train_loss,
        'test_loss': test_loss,
        'f1s_micro': f1s_micro,
        'f1s_macro': f1s_macro,
        'f1s_weighted': f1s_weighted
    }

    df = pd.DataFrame(data)

    df.to_csv('Results/' + experience_name + '.csv', index=False)


def hereo(row):
    t1 = row['history_text']
    t2 = row['powers_text']
    return 'History:' + t1 + '\n Powers:' + t2


def n(s):
    return (s - s.min()) / (s.max() - s.min())

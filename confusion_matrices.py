import sys
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from useful_functions import print_list


def myloss(labels, predictions, C, device):
    M = torch.zeros((C, C), device=device)
    for i in range(C):
        y = labels[i]
        yhat = predictions[i]

        y_1 = y / y.sum()
        yhat_1 = yhat / yhat.sum()
        min = torch.minimum(y_1, yhat_1)
        diag = torch.diag(min)
        if not torch.equal(y_1, yhat_1):
            off_diag = torch.outer(y_1 - min, yhat_1 - min) / (y_1 - min).sum()
        else:
            off_diag = torch.zeros_like(diag)
        M += (diag + off_diag)

    precision = torch.diag(M) / (M.sum(dim=1) + 1e-16)
    recall = torch.diag(M) / (M.sum(dim=0) + 1e-16)
    f1 = 2 * recall * precision / (recall + precision + 1e-16)
    return 1 - f1.mean()


def list2torch(l):
    if isinstance(l[0], list):
        return torch.tensor(l, dtype=torch.float32)
    else:
        C = len(l)
        return torch.tensor(l, dtype=torch.float32).reshape((1, C))


class confusion_matrix:
    def __init__(self, C, type, class_name=None):
        assert type in {'CM', 'TCMone', 'TCMlab', 'TCMpred', 'MLCM', 'SCMe', 'SCM_min', 'SCM_max', 'MLCTr', 'MLCTp'}

        if type == 'CM':
            self.operator = self.CM
        elif type == 'TCMone' or type == 'TCMlab' or type == 'TCMpred':
            self.operator = self.TCM
        elif type == 'MLCM':
            self.operator = self.MLCM
        elif type == 'SCMe':
            self.operator = self.SCM
        elif type == 'SCM_min':
            self.operator = self.SCM_min
        elif type == 'SCM_max':
            self.operator = self.SCM_max
        elif type == 'MLCTr':
            self.operator = self.MLCT_R
        elif type == 'MLCTp':
            self.operator = self.MLCT_P

        if class_name is None:
            self.class_name = ['c' + str(i) for i in range(C)]
        else:
            self.class_name = class_name

        self.C = C

        self.M = torch.zeros((C, C))

        if type == 'TCMlab':
            self.weighting = self.true_weighting
        elif type == 'TCMpred':
            self.weighting = self.pred_weighting
        elif type == 'TCMone':
            self.weighting = self.equal_weighting

    def true_weighting(self, y, yhat):
        return y.sum()

    def pred_weighting(self, y, yhat):
        return yhat.sum()

    def equal_weighting(self, y, yhat):
        return 1

    def CM(self, y, yhat):
        assert y.sum() == 1 and yhat.sum() == 1
        return torch.outer(y, yhat)

    def TCM(self, y, yhat):
        y_1 = y / y.sum()
        yhat_1 = yhat / yhat.sum()
        min = torch.minimum(y_1, yhat_1)
        diag = torch.diag(min)
        if not torch.equal(y_1, yhat_1):
            off_diag = torch.outer(y_1 - min, yhat_1 - min) / (y_1 - min).sum()
        else:
            off_diag = torch.zeros_like(diag)
        return self.weighting(y, yhat) * (diag + off_diag)

    def MLCM(self, y, yhat):
        min = torch.minimum(y, yhat)

        if torch.equal(y, yhat):
            return torch.diag(y)
        elif torch.all(y - min == 0):
            return torch.outer(y, yhat - min) / yhat.sum() + torch.diag(y) * y.sum() / yhat.sum()
        elif torch.all(yhat - min == 0):
            return torch.outer(y - min, yhat) / yhat.sum() + torch.diag(yhat)
        else:
            return torch.outer(y - min, yhat - min) / (yhat - min).sum() + torch.diag(min)

    def MLCT_R(self, y, yhat):
        min = torch.minimum(y, yhat)
        return torch.diag(min) + torch.outer(y - min, yhat) / yhat.sum()

    def MLCT_P(self, y, yhat):
        min = torch.minimum(y, yhat)
        return torch.diag(min) + torch.outer(y, yhat - min) / y.sum()

    def SCM_max(self, y, yhat):
        y_1 = y / y.sum()
        yhat_1 = yhat / yhat.sum()
        min = torch.minimum(y_1, yhat_1)
        r_prime = y_1 - min
        s_prime = yhat_1 - min

        return torch.diag(min) + torch.minimum(r_prime.repeat(self.C, 1).T, s_prime.repeat(self.C, 1))

    def SCM_min(self, y, yhat):
        y_1 = y / y.sum()
        yhat_1 = yhat / yhat.sum()
        min = torch.minimum(y_1, yhat_1)
        r_prime = y_1 - min
        s_prime = yhat_1 - min
        return torch.diag(min) + torch.maximum(r_prime.repeat(self.C, 1).T + s_prime.repeat(self.C, 1) - s_prime.sum(),
                                               torch.zeros((self.C, self.C))).fill_diagonal_(0)

    def SCM(self, y, yhat):
        y_1 = y / y.sum()
        yhat_1 = yhat / yhat.sum()
        min = torch.minimum(y_1, yhat_1)
        r_prime = y_1 - min
        s_prime = yhat_1 - min
        A = torch.diag(min) + torch.maximum(r_prime.repeat(self.C, 1).T + s_prime.repeat(self.C, 1) - s_prime.sum(),
                                            torch.zeros((self.C, self.C))).fill_diagonal_(0)
        B = torch.diag(min) + torch.minimum(r_prime.repeat(self.C, 1).T, s_prime.repeat(self.C, 1))
        return (A + B) / 2

    def update(self, labels, predictions):
        assert isinstance(labels, torch.Tensor) and ((labels.dtype == torch.float32) or (labels.dtype == torch.double))
        assert isinstance(predictions, torch.Tensor) and (
                (predictions.dtype == torch.float32) or (predictions.dtype == torch.double))

        for i in range(len(predictions)):
            y = labels[i]
            yhat = predictions[i]
            self.M += self.operator(y, yhat)

    def score(self):
        precision = torch.diag(self.M) / self.M.sum(dim=1)
        recall = torch.diag(self.M) / self.M.sum(dim=0)
        return 2 * recall * precision / (recall + precision)

    def get(self, normalisation='raw'):
        assert normalisation in {'true', 'pred', 'all', 'raw'}

        if normalisation == 'true':
            return self.M / self.M.sum(dim=1, keepdims=True)
        elif normalisation == 'pred':
            return self.M / self.M.sum(dim=0, keepdims=True)
        elif normalisation == 'all':
            return self.M / self.M.sum()
        else:
            return self.M

    def print(self, normalization='raw', class_name=None, diag=False):
        assert normalization in {'true', 'pred', 'all', 'raw', '100'}

        pd.set_option('display.float_format', lambda x: '%.1f' % x)

        M = self.M.numpy()

        if diag:
            if normalization == 'true':
                print_list(np.diag(100 * M / M.sum(axis=1, keepdims=True)))
            elif normalization == 'pred':
                print_list(np.diag(100 * M / M.sum(axis=0, keepdims=True)))
            elif normalization == 'all':
                print_list(np.diag(100 * M / M.sum()))
            elif normalization == '100':
                print_list(np.diag(100 * M))
            else:
                print_list(np.diag(M))
        else:
            if normalization == 'true':
                print(pd.DataFrame(100 * M / M.sum(axis=1, keepdims=True), columns=self.class_name,
                                   index=self.class_name).to_string())
            elif normalization == 'pred':
                print(pd.DataFrame(100 * M / M.sum(axis=0, keepdims=True), columns=self.class_name,
                                   index=self.class_name).to_string())
            elif normalization == 'all':
                print(pd.DataFrame(100 * M / M.sum(), columns=self.class_name,
                                   index=self.class_name).to_string())
            elif normalization == '100':
                print(pd.DataFrame(100 * M, columns=self.class_name,
                                   index=self.class_name).to_string())
            else:
                print(pd.DataFrame(M, columns=class_name,
                                   index=class_name).to_string())

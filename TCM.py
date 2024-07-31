import sys
import torch
import numpy as np
import random

"""
The class TCM is an implementation of TCM with no loops in the code, enabling faster execution speed.

The class TCM_loop_in_code is an implementation of TCM with loops in the code, resulting in more intuitive code, but slower execution speed.

The first part of the code (I) shows that the two classes return the same matrices.

The second part of the code (II) is a simple example of use.
"""


class TCM:
    def __init__(self, C, type):
        self.M = torch.zeros((C, C))  # matrix incremented at each batch
        self.C = C  # number of classes

        assert type in ['TCMlab', 'TCMpred', 'TCMone']
        self.type = type  # weighting

    def update(self, labels, predictions):
        # verification that labels and predictions are numpy matrix or pytorch matrix of the expectated shape batch size * number of classes ie n * C
        if isinstance(labels, np.ndarray) and isinstance(predictions, np.ndarray):
            labels, predictions = torch.from_numpy(labels).type(torch.DoubleTensor), torch.from_numpy(predictions).type(
                torch.DoubleTensor)
        elif isinstance(labels, torch.Tensor) and isinstance(predictions, torch.Tensor):
            if ((labels.dtype != torch.float32) and (labels.dtype != torch.double)) or (
                    (predictions.dtype != torch.float32) and (predictions.dtype != torch.double)):
                labels, predictions = labels.type(torch.float32), predictions.type(torch.float32)
        else:
            raise Exception('Only numpy and torch types are accepted.')

        assert labels.shape == predictions.shape
        n, C = labels.shape
        assert C == self.C

        norm_pred = predictions.sum(axis=1, keepdim=True)  # norm of each prediction row
        norm_lab = labels.sum(axis=1, keepdim=True)  # norm of each label row

        # weighting
        if self.type == 'TCMone':
            weights = torch.ones(n).unsqueeze(1)
        elif self.type == 'TCMpred':
            weights = norm_pred
        elif self.type == 'TCMlab':
            weights = norm_lab

        normalized_pred = predictions / norm_pred  # normalized prediction
        normalized_lab = labels / norm_lab  # normalized label

        common_quantities = torch.minimum(normalized_pred,
                                          normalized_lab)  # common quantities between normalized prediction and label

        weighted_common_quantities = weights * common_quantities  # weighted common quantities

        diag = torch.diag(weighted_common_quantities.sum(axis=0))  # diagonal matrix corresponding of diagonal of TCM

        lab_error = normalized_lab - common_quantities  # remove to each label row common quantities
        pred_error = normalized_pred - common_quantities  # remove to each prediction row common quantities

        errors = (predictions != labels).any(
            dim=1).bool()  # vector with 0 if prediction and label match perfectly 1 otherwise
        lab_error = lab_error[errors, :]  # considering only label associated with unperfect prediction
        pred_error = pred_error[errors, :]  # considering only unperfect prediction
        error_weights = torch.sqrt(weights[errors, :])  # considering only weight associated with nperfect prediction

        # weighted each row as expected
        sqrt_denominator = torch.sqrt(lab_error.sum(axis=1, keepdim=True))
        lab_error *= error_weights / sqrt_denominator
        pred_error *= error_weights / sqrt_denominator

        # outer product of each row
        lab_error_unsqueezed = lab_error.unsqueeze(2)
        pred_error_unsqueezed = pred_error.unsqueeze(1)
        off_diag = (lab_error_unsqueezed * pred_error_unsqueezed).sum(axis=0)

        self.M += diag + off_diag

    def get(self):
        return self.M


class TCM_loop_in_code:
    def __init__(self, C, type):
        self.M = torch.zeros((C, C))  # matrix incremented at each batch
        self.C = C  # number of classes

        # weighting
        assert type in ['TCMlab', 'TCMpred', 'TCMone']
        if type == 'TCMone':
            self.weighting = self.one_weighting
        elif type == 'TCMpred':
            self.weighting = self.pred_weighting
        elif type == 'TCMlab':
            self.weighting = self.lab_weighting

    def one_weighting(self, y, yhat):
        return 1

    def lab_weighting(self, y, yhat):
        return y.sum()

    def pred_weighting(self, y, yhat):
        return yhat.sum()

    def update(self, labels, predictions):
        # verification that labels and predictions are numpy matrix or pytorch matrix of the expectated shape batch size * number of classes ie n * C
        if isinstance(labels, np.ndarray) and isinstance(predictions, np.ndarray):
            labels, predictions = torch.from_numpy(labels).type(torch.DoubleTensor), torch.from_numpy(predictions).type(
                torch.DoubleTensor)
        elif isinstance(labels, torch.Tensor) and isinstance(predictions, torch.Tensor):
            if ((labels.dtype != torch.float32) and (labels.dtype != torch.double)) or (
                    (predictions.dtype != torch.float32) and (predictions.dtype != torch.double)):
                labels, predictions = labels.type(torch.float32), predictions.type(torch.float32)
        else:
            raise Exception('Only numpy and torch types are accepted.')

        assert labels.shape == predictions.shape
        n, C = labels.shape
        assert C == self.C

        # loop computing each contribution
        for i in range(n):
            y = labels[i]
            yhat = predictions[i]
            self.M += self.contribution(y, yhat)

    def contribution(self, y, yhat):
        y_1 = y / y.sum()  # normalized label
        yhat_1 = yhat / yhat.sum()  # normalized prediction
        min = torch.minimum(y_1, yhat_1)  # common quantities
        diag = torch.diag(min)  # diagonal matrix corresponding of diagonal of the contribution

        if not torch.equal(y_1, yhat_1):  # if an error
            off_diag = torch.outer(y_1 - min, yhat_1 - min) / (y_1 - min).sum()
        else:  # if perfect match no off diagonal terms
            off_diag = torch.zeros_like(diag)

        # return weighted contribution
        return self.weighting(y, yhat) * (diag + off_diag)

    def get(self):
        return self.M


"""
I) Showing that the two classes return the same matrices
"""
# random data
n = 30  # batch size
C = 5  # number of class
pred = torch.rand(n, C)  # prediction
lab = torch.rand(n, C)  # label

# randomly setting equal rows in label and prediction corresponding to perfect match
numbers = list(range(30))
random.shuffle(numbers)
perfect_match = torch.tensor(numbers[:5])
pred[perfect_match, :] = lab[perfect_match, :]

# test that the two classes return the same matrices
for type in ['TCMone', 'TCMlab', 'TCMpred']:
    M_no_loop = TCM(C, type)
    M_no_loop.update(lab, pred)

    M_loop = TCM_loop_in_code(C, type)
    M_loop.update(lab, pred)

    if torch.allclose(M_loop.get(), M_no_loop.get()):
        print(type, 'are equal')

print()
"""
II) A simple example of use 
"""
# random data
n = 30  # batch size
C = 4  # number of class
type = 'TCMlab'  # weighting
M = TCM(C, type)

# data
batched_pred = [torch.rand(n, C), torch.rand(n, C), torch.rand(n, C)]  # batches of prediction
batched_lab = [torch.rand(n, C), torch.rand(n, C), torch.rand(n, C)]  # batches of label

# go through all the batches and update the matrix on each of them
for pred, lab in zip(batched_pred, batched_lab):
    M.update(lab, pred)

# get TCMlab
print(M.get())

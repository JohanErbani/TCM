import numpy as np
import torch

class TCM:
    def __init__(self, C):
        self.M = torch.zeros((C, C))

    def update(self, labels, predictions):
        assert isinstance(labels, torch.Tensor) and ((labels.dtype == torch.float32) or (labels.dtype == torch.double))
        assert isinstance(predictions, torch.Tensor) and (
                    (predictions.dtype == torch.float32) or (predictions.dtype == torch.double))
        n = len(labels)

        for i in range(n):
            y = labels[i]
            weight = y.sum()
            y_dist = y/y.sum()
            yhat = predictions[i]
            yhat_dist = yhat/yhat.sum()
            min = torch.minimum(y_dist, yhat_dist)
            diag = torch.diag(min)
            if not torch.equal(y_dist, yhat_dist):
                off_diag = torch.outer(y_dist - min, yhat_dist - min)/(y_dist - min).sum()
            else:
                off_diag = torch.zeros_like(diag)
            self.M += (diag + off_diag)

    def get(self):
        return self.M


labels=torch.tensor([[1,1,0],[1,1,0]])
predictions=torch.tensor([[2,1,0],[1,1,0]])

M = TCM(3)
M.update(labels, predictions)
print(M.get())
print(labels==predictions)
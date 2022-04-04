from layers import HNN, LinearDecoder
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
import torch.nn.functional as F

def acc_f1(output, labels, average='binary'):
    preds = output.max(1)[1].type_as(labels)
    if preds.is_cuda:
        preds = preds.cpu()
        labels = labels.cpu()
    accuracy = accuracy_score(preds, labels)
    f1 = f1_score(preds, labels, average=average)
    return accuracy, f1

class NCModel(nn.Module):
    """
    Base model for node classification task.
    """

    def __init__(self):
        super(NCModel, self).__init__()
        self.c = torch.tensor([1.0])
        self.decoder = LinearDecoder(self.c)
        self.encoder = HNN()
        self.f1_average = 'micro'
        self.weights = torch.Tensor([1.] * 3)
    
    def encode(self, x, adj):
        h = self.encoder.encode(x, adj)
        return h

    def decode(self, h, adj, idx):
        output = self.decoder.decode(h, adj)
        return F.log_softmax(output[idx], dim=1)

    def compute_metrics(self, embeddings, data, split):
        idx = data[f'idx_{split}']
        output = self.decode(embeddings, data['adj_train_norm'], idx)
        loss = F.nll_loss(output, data['labels'][idx], self.weights)
        acc, f1 = acc_f1(output, data['labels'][idx], average=self.f1_average)
        metrics = {'loss': loss, 'acc': acc, 'f1': f1}
        return metrics

    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1["f1"] < m2["f1"]
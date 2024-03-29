import torch
from sklearn.metrics import accuracy_score, f1_score


class Evaluate:

    def __init__(self, metric_name):
        self.metric_name = metric_name
        self.predictions = None
        self.references = None

    def add_batch(self, predictions, references):
        predictions = predictions.detach().cpu()
        references = references.detach().cpu()
        self.predictions = predictions if self.predictions is None \
            else torch.cat([self.predictions, predictions], dim=0)
        self.references = references if self.references is None else torch.cat([self.references, references], dim=0)
    
    @staticmethod
    def _keep_four_digits(number):
        return float('{:.4f}'.format(number))
    
    def compute(self, average='macro'):
        if self.metric_name == 'accuracy':
            return {'accuracy': self._keep_four_digits(accuracy_score(self.references, self.predictions))}
        elif self.metric_name == 'f1':
            return {'f1': self._keep_four_digits(f1_score(self.references, self.predictions, average=average))}


def load(metric_name):
    if metric_name in ['accuracy', 'f1']:
        return Evaluate(metric_name)
    else:
        raise ValueError(f'Unknown metric name {metric_name}')
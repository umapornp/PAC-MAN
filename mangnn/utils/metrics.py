import torch
import torch.nn.functional as F


class MetricCaller:
    """ Metrics caller. """

    def __init__(self, metrics):
        """ Constructor for `MetricsCaller`. 
        
        Args:
            metrics: List of metrics.
        """
        self.metrics = metrics


    def update(self, logits, labels):
        """ Update logits and labels to all metrics. 
        
        Args:
            logits: Logits.
            labels: Labels.
        """
        for metric in self.metrics:
            metric.update(logits, labels)


    def compute(self):
        """ Compute all metrics. 
        
        Returns:
            metrics: Dictionary of all metrics.
        """
        metrics = {}

        for metric in self.metrics:
            metric.compute()
            metrics.update(metric.get())
        return metrics


    def get(self):
        """ Get all metrics.
        
        Returns:
            metrics: Dictionary of all metrics.
        """
        metrics = {}
        for metric in self.metrics:
            metrics.update(metric.get())
        return metrics


    def reset(self):
        """ Reset values in all metrics. """
        for metric in self.metrics:
            metric.reset()


class Metric:
    """ Metric. """

    def __init__(self, name=None):
        """ Constructor for `Metric`. 
        
        Args:
            name: Metric name.
        """
        self.name = name
        self.value = None
        self.logits = None
        self.labels = None


    def update(self, logits, labels):
        """ Update logits and labels to metric.

        Args:
            logits: Logits. [batch_size, 2]
            labels: Labels. [batch_size]
        """
        # Perform argmax().
        if logits.shape != labels.shape:
            logits = logits.argmax(dim=-1)

        # Update current logits and labels.
        self.logits = nested_concat(self.logits, logits, pad_value=0)
        self.labels = nested_concat(self.labels, labels, pad_value=-1)


    def compute(self):
        """ Compute metric. """
        pass


    def get(self):
        """ Get metric.
        
        Returns:
            Dictionary of metric. 
        """
        if isinstance(self.value, dict):
            return {k:v.item() for k,v in self.value.items()}
        else:
            return {self.name: self.value.item()}


    def reset(self):
        """ Reset metric. """
        self.logits = None
        self.labels = None
        self.value = None


class ClassificationReport(Metric):
    """ Classification report: accuracy, precision, recall, and f1-score. """
    
    def __init__(self):
        """ Constructor for `ClassificationReport`. """
        super().__init__(name="classification_report")

    
    def compute(self):
        """ Compute classification report. """
        
        # Compute confusion matrix.
        tp = ((self.labels == self.logits) & (self.labels == 1)).sum()
        tn = ((self.labels == self.logits) & (self.labels == 0)).sum()
        fp = ((self.labels != self.logits) & (self.labels == 0)).sum()
        fn = ((self.labels != self.logits) & (self.labels == 1)).sum()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        if precision != 0 or recall != 0:
            f1_score = (2 * precision * recall) / (precision + recall)
        else:
            f1_score = torch.tensor(0.)
        
        support = tp + fn

        self.value = {"accuracy": accuracy, 
                    "precision": precision,
                    "recall": recall,
                    "f1-score": f1_score, 
                    "support": support}

        return self.value


def nested_concat(tensor_a, tensor_b, pad_value=0):
    """ Nested concatenation between two tensors. 
    
    Args:
        tensor_a: Tensor A.
        tensor_b: Tensor B.
        pad_value: Padding value in case size of two tensors are different. (Default=`0`).
        
    Returns:
        Concatenated tensor.
    """
    if  tensor_a is None:
        tensor_a = tensor_b
    else:
        # Padding
        if tensor_b.shape[-1] != tensor_a.shape[-1]:
            tensor_a, tensor_b = pad(tensor_a, tensor_b, pad_value)

        # Concatenate
        tensor_a = torch.cat((tensor_a, tensor_b), dim=0)

    return tensor_a


def pad(a, b, value=0):
    """ Padding tensor.
    
    Args:
        a,b: Tensor.
        value: Padding value (Default=`0`).
    
    Returns:
        Tensor with padding value.
    """
    max_len = max(a.shape[-1], b.shape[-1])
    a = F.pad(a, (0, max_len - a.shape[-1]), value=value)
    b = F.pad(b, (0, max_len - b.shape[-1]), value=value)
    return a, b
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
            logits: Logits.
            labels: Labels.
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
        return {self.name: self.value.item()}


    def reset(self):
        """ Reset metric. """
        self.logits = None
        self.labels = None
        self.value = None


class Precision(Metric):
    """ Precision@k """

    def __init__(self, num_classes, topk=None):
        """ Constructor for `Precision`. 
        
        Args:
            num_classes: Number of class labels. 
            topk: Top-k.
        """
        super().__init__(name=f"p@{topk}" if topk else "p")
        self.num_classes = num_classes
        self.topk = topk


    def compute(self):
        """ Compute precision@k. 
        
        Returns:
            Precision@k.
        """
        true_pos = compute_true_positive(self.logits, self.labels, self.num_classes)
        total_pred = (self.labels > -1).sum(dim=1)

        # Select top-k.
        if self.topk:
            true_pos, _ = select_topk(true_pos, self.labels, self.topk)
            total_pred = self.topk

        # Compute.
        self.value = (true_pos.sum(dim=1) / total_pred).nanmean()

        return self.value


class Recall(Metric):
    """ Recall@k """

    def __init__(self, num_classes, topk=None):
        """ Constructor for `Recall`. 
        
        Args:
            num_classes: Number of class labels.
            topk: Top-k.
        """
        super().__init__(name=f"r@{topk}" if topk else "r")
        self.num_classes = num_classes
        self.topk = topk


    def compute(self):
        """ Compute recall@k 
        
        Returns:
            Recall@k.
        """
        true_pos = compute_true_positive(self.logits, self.labels, self.num_classes)
        total_labels = (self.labels > -1).sum(dim=1)

        # Select top-k.
        if self.topk:
            true_pos, total_labels = select_topk(true_pos, self.labels, self.topk)
            total_labels = (total_labels > -1).sum(dim=1)

        # Compute.
        self.value = (true_pos.sum(dim=1) / total_labels).nanmean()

        return self.value


class F1Score(Metric):
    """ F1-Score@k. """

    def __init__(self, num_classes, topk=None):
        """ Constructor for `F1Score`. 
        
        Args:
            num_classes: Number of class labels.
            topk: Top-k.
        """
        super().__init__(name=f"f1@{topk}" if topk else "f1")
        self.num_classes = num_classes
        self.topk = topk
        self.precision = Precision(num_classes=self.num_classes, topk=self.topk)
        self.recall = Recall(num_classes=self.num_classes, topk=self.topk)


    def compute(self):
        """ Compute f1-score@k.
        
        Returns:
            F1-Score@k
        """
        # Update logits and labels to precision and recall.
        self.precision.logits = self.logits
        self.precision.labels = self.labels
        self.recall.logits = self.logits
        self.recall.labels = self.labels

        # Compute precision and recall.
        precision = self.precision.compute()
        recall = self.recall.compute()

        # Compute f1-score.
        if precision != 0 or recall != 0:
            self.value = 2 * precision * recall / (precision + recall)
        else:
            self.value = torch.tensor(0.)

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


def create_onehot(tensor, index, num_classes):
    """ Create onehot tensor.
    
    Args:
        tensor: Input tensor.
        index: Index of class labels.
        num_classes: Number of class labels.
    
    Returns:
        onehot: Onehot tensor.
    """
    onehot = tensor.new_zeros((tensor.shape[0], num_classes))
    onehot[index[0], tensor[index]] = 1
    return onehot


def compute_true_positive(logits, labels, num_classes):
    """ Compute true positive. 
    
    Args:
        logits: Logits.
        labels: Labels.
        num_classes: Number of class labels.
    
    Returns:
        true_pos: True positive.
    """
    mask_idx = (labels > -1).nonzero(as_tuple=True)
    true_pos = logits.new_zeros(logits.shape)
    labels_onehot = create_onehot(labels, mask_idx, num_classes)
    correct = labels_onehot[mask_idx[0], logits[mask_idx]]
    true_pos[mask_idx] = correct
    return true_pos


def select_topk(true_pos, labels, k):
    """ Select top-k.
    
    Args:
        true_pos: Number of true positive values.
        labels: Labels.
        k: Value of k.
    
    Returns:
        true_pos: Top-k selected true positive.
        labels: Top-k selected labels.
    """
    # Filter out rows having masks < k.
    keep_row = (labels > -1).sum(dim=1) >= k 
    true_pos = true_pos[keep_row]
    labels = labels[keep_row]

    # Select top-k logits.
    _, idx = (labels > -1).sort(dim=-1, descending=True, stable=True)
    true_pos = true_pos.gather(dim=1, index=idx)
    true_pos = true_pos[:,:k]

    return true_pos, labels
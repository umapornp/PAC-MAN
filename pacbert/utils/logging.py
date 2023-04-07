import logging
import torch


# Logging format.
_LOG_FMT = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s'
_DATE_FMT = '%m/%d/%Y %H:%M:%S'

# Set global logging.
logging.basicConfig(format=_LOG_FMT, datefmt=_DATE_FMT, level=logging.INFO)
LOG = logging.getLogger('__main__')


class TrainLogger:
    """ Log value over training. """

    def __init__(self, name):
        """ Constructor for `TrainLogger`. 
        
        Args:
            name: Name of value for logging.
        """
        self.name = name
        self.value = None


    def update(self, value):
        """ Update value. 
        
        Args:
            value: Value for logging.
        """
        value = torch.tensor([value])

        if self.value is None:
            self.value = value
        else:
            self.value = torch.cat((self.value, value))


    def get(self):
        """ Get average value.
        
        Returns:
            Average value.
        """
        return self.value.nanmean().item()


    def reset(self):
        """ Reset value. """
        self.value = None


def print_tensor(tensor, name, value=True):
    """ Print tensor.
    
    Args:
        tensor: Tensor.
        name: Tensor name.
        value: Whether to print value in tensor (Default=`True`).
    
    """
    print("="*70)
    print(f"Tensor: {name}")
    print("-"*70)
    if tensor is None:
        print(f"Shape: {tensor}")
    else:
        print(f"Shape: {tensor.shape}")
    if value:
        print(f"{tensor}")
    print(f"="*70)
    print()


def print_list(list, name, value=True):
    """ Print list or tuple.
    
    Args:
        list: List.
        name: List name.
        value: Whether to print value in list (Default=`True`).
    """
    print("="*70)
    print(f"List: {name}")
    print("-"*70)
    print(f"Shape: {len(list)}")
    if value:
        print(f"{list}")
    print(f"="*70)
    print()
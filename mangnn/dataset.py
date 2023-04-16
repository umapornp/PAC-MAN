import numpy as np
from torch.utils.data import Dataset


DATA_ROOT = {"twitter": "mangnn/data/twitter"}


DATA_PATH = {
    "twitter": {
        "train": f"{DATA_ROOT['twitter']}/twitter_train.npy",
        "val": f"{DATA_ROOT['twitter']}/twitter_val.npy",
        "test": f"{DATA_ROOT['twitter']}/twitter_test.npy",
    }
}


class TwitterDataset(Dataset):
    """Twitter dataset."""

    def __init__(self, split="train", data_path=None):
        """Constructor for `TwitterDataset`.

        Args:
            split: Type of dataset (`train`, `val`, and `test`).
            data_path: Data path.
        """
        super(TwitterDataset, self).__init__()
        self.split = split
        self.data_path = data_path
        if self.data_path is None:
            self.data_path = DATA_PATH["twitter"][self.split]
        self._dataset = None


    @property
    def dataset(self):
        """Return the dataset."""
        if self._dataset is None:
            self._build_dataset()
        return self._dataset


    def _build_dataset(self):
        """Read the dataset from file.

        Returns:
            dataset: Dictionary of dataset.
        """
        self._dataset = np.load(self.data_path, allow_pickle=True)
        return self._dataset


    def __getitem__(self, idx):
        """Prepare and return dataset entry.

        Args:
            idx: Index of the dataset to return.

        Returns:
            Tuple of inputs and labels.
            * inputs: Dictionary of user_ids and tag_ids.
            * labels: Labels (0 and 1).
        """
        user_ids, tag_ids, labels = self.dataset[idx]
        return ({"user_ids": user_ids, "tag_ids": tag_ids}, labels)


    def __len__(self):
        """Return the number of dataset."""
        return len(self.dataset)
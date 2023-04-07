import torch

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence


class TwitterDataModule:
    """ Twitter data module. """

    def __init__(self, batch_size, num_workers, use_ddp):
        """ Constructor for `TwitterDataModule`.
        
        Args:
            batch_size: Batch size.
            num_workers: Number of workers for dataloader.
            use_ddp: Whether to use DDP.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_ddp = use_ddp

        # Datasets
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

        # Samplers
        self._train_sampler = None
        self._val_sampler = None
        self._test_sampler = None

        # Data loaders
        self._train_loader = None
        self._val_loader = None
        self._test_loader = None


    @property
    def train_dataset(self):
        """ Return training dataset. """
        return self._train_dataset


    @property
    def val_dataset(self):
        """ Return validation dataset. """
        return self._val_dataset


    @property
    def test_dataset(self):
        """ Return testing dataset. """
        return self._test_dataset


    @train_dataset.setter
    def train_dataset(self, dataset):
        """" Set training dataset.
        
        Args:
            dataset: Training dataset. 
        """
        self._train_dataset = dataset


    @val_dataset.setter
    def val_dataset(self, dataset):
        """ Set validation dataset.
        
        Args:
            dataset: Validation dataset.
        """
        self._val_dataset = dataset


    @test_dataset.setter
    def test_dataset(self, dataset):
        """ Set testing dataset.
        
        Args:
            dataset: Testing dataset.
        """
        self._test_dataset = dataset


    @property
    def train_sampler(self):
        """ Return training sampler. """
        if self._train_sampler is None:
            self._build_train_sampler()
        return self._train_sampler


    @property
    def val_sampler(self):
        """ Return validation sampler. """
        if self._val_sampler is None:
            self._build_val_sampler()
        return self._val_sampler


    @property
    def test_sampler(self):
        """ Return testing sampler. """
        if self._test_sampler is None:
            self._build_test_sampler()
        return self._test_sampler


    @property
    def train_loader(self):
        """ Return training dataloader. """
        if self._train_loader is None:
            self._build_train_loader()
        return self._train_loader


    @property
    def val_loader(self):
        """ Return validation dataloader. """
        if self._val_loader is None:
            self._build_val_loader()
        return self._val_loader


    @property
    def test_loader(self):
        """ Return testing dataloader. """
        if self._test_loader is None:
            self._build_test_loader()
        return self._test_loader


    def _build_train_sampler(self):
        """ Build training sampler. """
        if self.use_ddp:
            self._train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
        else:
            self._train_sampler = RandomSampler(self.train_dataset)

        return self._train_sampler


    def _build_val_sampler(self):
        """ Build validation sampler. """
        if self.use_ddp:
            self._val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
        else:
            self._val_sampler = SequentialSampler(self.val_dataset)

        return self._val_sampler


    def _build_test_sampler(self):
        """ Build testing sampler. """
        if self.use_ddp:
            self._test_sampler = DistributedSampler(self.test_dataset, shuffle=False)
        else:
            self._test_sampler = SequentialSampler(self.test_dataset)

        return self._test_sampler


    def _build_train_loader(self):
        """ Build training dataloader. """
        self._train_loader = DataLoader(dataset=self.train_dataset,
                                        sampler=self.train_sampler,
                                        batch_size=self.batch_size,
                                        num_workers=self.num_workers,
                                        persistent_workers=False,
                                        pin_memory=True,
                                        collate_fn=self._collate_fn)
        return self._train_loader
    

    def _build_val_loader(self):
        """ Build validation dataloader. """
        self._val_loader = DataLoader(dataset=self.val_dataset,
                                    sampler=self.val_sampler,
                                    batch_size=self.batch_size,
                                    num_workers=self.num_workers,
                                    pin_memory=True,
                                    collate_fn=self._collate_fn)
        return self._val_loader


    def _build_test_loader(self):
        """ Build testing dataloader. """
        self._test_loader = DataLoader(dataset=self.test_dataset,
                                    sampler=self.test_sampler,
                                    batch_size=self.batch_size,
                                    num_workers=self.num_workers,
                                    pin_memory=True,
                                    collate_fn=self._collate_fn)
        return self._test_loader


    def _collate_fn(self, batch):
        """ Collate function for preparing batch.
        
        Args:
            batch: Batch data.
        
        Returns:
            Tuple of:
            * inputs: Dictionary of user_ids, text_ids, and tag_ids
            * labels: Tag labels.
        """
        user_ids, text_ids, tag_ids, labels = [], [], [], []

        for _, (inp, lab) in enumerate(batch):
            user_ids.append(torch.tensor(inp["user_ids"]))
            text_ids.append(torch.tensor(inp["text_ids"]))
            tag_ids.append(torch.tensor(inp["tag_ids"]))
            labels.append(torch.tensor(lab))

        # Padding
        user_ids = torch.stack(user_ids)
        text_ids = pad_sequence(text_ids, batch_first=True, padding_value=0)
        tag_ids = pad_sequence(tag_ids, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-1)

        return ({"user_ids": user_ids,
                "text_ids": text_ids,
                "tag_ids": tag_ids}, labels)
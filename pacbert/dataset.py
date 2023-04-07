import json
import logging
import random

from torch.utils.data import Dataset
from transformers import BertTokenizer
from pacbert.utils.tagtokenizer import TagTokenizer


DATA_ROOT = {
    "twitter": "pacbert/data/twitter"
}


DATA_PATH = {
    "twitter":{
        "train": f"{DATA_ROOT['twitter']}/twitter_train.json",
        "val": f"{DATA_ROOT['twitter']}/twitter_val.json",
        "test": f"{DATA_ROOT['twitter']}/twitter_test.json"
    }
}


class TwitterDataset(Dataset):
    """ Twitter dataset. """

    def __init__(self, split='train', data_path=None):
        """ Constructor for `TwitterDataset`.
        
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

        # Use pretrained BertTokenizer from huggingface library.
        self.text_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tag_tokenizer = TagTokenizer()


    @property
    def dataset(self):
        """ Return the dataset. """
        if self._dataset is None:
            self._build_dataset()
        return self._dataset


    def _build_dataset(self):
        """ Read the dataset from file.
        
        Returns:
            dataset: Dictionary of dataset.
        """
        f = open(self.data_path)
        self._dataset = json.load(f)
        f.close()
        return self._dataset


    def get_vocab_size(self):
        """ Return vocabulary size. """
        return self.text_tokenizer.vocab_size


    def get_tag_size(self):
        """ Return tag size. """
        return self.tag_tokenizer.vocab_size


    def __getitem__(self, idx):
        """ Prepare and return dataset entry.

        Args:
            idx: Index of the dataset to return.

        Returns:
            Tuple of inputs and labels.
            * inputs: Dictionary of user_ids, text_ids, and tag_ids.
            * labels: Tuple of masked tag labels.
        """
        data = self.dataset[idx]

        user_ids = [data['user']]
        text = data['text']
        tag = data['tag']

        # Prepare text.
        text_tokens = self.text_tokenizer.tokenize(text)
        text_tokens = ['[CLS]'] + text_tokens + ['[SEP]']
        text_ids = self.text_tokenizer.convert_tokens_to_ids(text_tokens)

        # Prepare tag.
        # Randomly mask for training and entirely mask for validation.
        mask_mode = "random" if self.split == "train" else "all"
        tag_ids, labels = self._mask_tag(tag, mask_mode)

        return ({"user_ids": user_ids,
                "text_ids": text_ids,
                "tag_ids": tag_ids}, labels)


    def __len__(self):
        """ Return the number of dataset. """
        return len(self.dataset)


    def _mask_tag(self, tags, mode):
        """ Mask tags.
        
        Args:
            tags: List of tags to mask.
            mode: Mask mode (`all` and `random`).
        """
        if mode == "all":
            return self._mask_all(tags, self.tag_tokenizer)
        elif mode == "random":
            return self._random_mask(tags, self.tag_tokenizer)


    def _random_mask(self, tokens, tokenizer):
        """ Randomly mask the tokens. 
        
        Args:
            tokens: Tokens to mask.
            tokenizer: Tokenizer.
        
        Returns:
            new_tokens: Masked tokens.
            labels: Labels of masked tokens.
        """
        labels = []
        new_tokens = []

        for i, token in enumerate(tokens):
            prob = random.random()

            # Mask token with 50% probability.
            if prob < 0.5:
                prob /= 0.5

                # 80% randomly change token to mask token.
                if prob < 0.8:
                    new_tokens.append(tokenizer.vocab["[MASK]"])

                # 10% randomly change token to random token.
                elif prob < 0.9:
                    new_tokens.append(random.choice(list(tokenizer.vocab.values())))
                
                # Rest 10% randomly keep current token.
                else:
                    new_tokens.append(tokenizer.vocab[token])

                # Append current token to output (we will predict these later).
                try:
                    labels.append(tokenizer.vocab[token])
                except KeyError:
                    # For unknown words.
                    labels.append(tokenizer.vocab["[UNK]"])
                    logging.warning("Cannot find token '{}' in vocab. Using [UNK] instead".format(token))
            else:
                # No masking token [-1] (will be ignored by loss function later).
                new_tokens.append(tokenizer.vocab[token])
                labels.append(-1)

        return new_tokens, labels


    def _mask_all(self, tokens, tokenizer):
        """ Entirely mask the tokens.
        
        Args:
            tokens: Tokens to mask.
            tokenizer: Tokenizer.
            
        Returns:
            tokens: Masked tokens.
            labels: Labels of masked tokens.
        """
        labels = tokenizer.convert_tags_to_ids(tokens)
        tokens = [tokenizer.vocab["[MASK]"]] * len(tokens)
        return tokens, labels

import collections

# Path to the tag file.
TAG_FILE = "pacbert/data/twitter/tag.txt"


def load_tag(tag_file):
    """ Load all tags from the tag file.
    
    Args:
        tag_file: Path to the tag file.
    
    Returns:
        tag2id: Dictionary mapping from tag to ID. 
        id2tag: Dictionary mapping from ID to tag.
    """
    tag2id = collections.OrderedDict()
    id2tag = collections.OrderedDict()

    with open(tag_file, "r", encoding="utf-8") as reader:
        tags = reader.readlines()

    for id, tag in enumerate(tags):
        tag = tag.rstrip("\n")
        tag2id[tag] = id
        id2tag[id] = tag
    return tag2id, id2tag


class TagTokenizer:
    """ Tag tokenizer. """

    def __init__(self, tag_file=None):
        """ Constructor for `TagTokenizer`. 
        
        Args:
            tag_file: Path to the tag file.
        """
        if tag_file is None:
            tag_file = TAG_FILE
        self.vocab, self.id2tag = load_tag(tag_file)


    @property
    def vocab_size(self):
        """ Return tag size. """
        return len(self.vocab)


    def get_vocab(self):
        """ Return dictionary mapping of tags. """
        return dict(self.vocab)


    def _convert_tag_to_id(self, tag):
        """ Convert tag to ID. 
        
        Args: 
            tag: Tag.
        
        Returns:
            id: Tag ID.
        """
        id = self.vocab.get(tag)
        return id


    def _convert_id_to_tag(self, id):
        """" Convert ID to tag.
        
        Args:
            id: Tag ID.
        
        Returns:
            tag: Tag.
        """
        tag = self.id2tag.get(id)
        return tag


    def convert_tags_to_ids(self, tags):
        """ Convert list of tags to list of IDs.
        
        Args:
            tags: List of tags.
        
        Returns:
            ids: List of tag IDs.
        """
        if isinstance(tags, str):
            return self._convert_tag_to_id(tags)

        ids = []
        for tag in tags:
            ids.append(self._convert_tag_to_id(tag))
        return ids


    def convert_ids_to_tags(self, ids):
        """" Convert list of IDs to list of tags. 
        
        Args:
            ids: List of tag IDs.

        Returns:
            tags: List of tags.
        """
        if isinstance(ids, int):
            return self._convert_id_to_tag(ids)
        
        tags = []
        for id in ids:
            id = int(id)
            tags.append(self._convert_id_to_tag(id))
        return tags
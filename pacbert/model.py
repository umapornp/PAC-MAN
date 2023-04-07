import torch
import torch.nn as nn

from pacbert.utils.logging import LOG
from pacbert.utils.modeling_bert import BertEncoder, BertPooler, BertPredictionHeadTransform
from pacbert.utils.tagtokenizer import TagTokenizer
from transformers import BertTokenizer, BertModel


def vocab_tag_mapping(model_name):
    """ Mapping between the same word and tag. 
    
    Args:
        model_name: Name of Bert pretrained model.
    
    Returns:
        vocab2tag: Dictionary mapping from word ID to tag ID.
        tag2vocab: Dictionary mappint from tag ID to word ID.
    """
    vocab2tag, tag2vocab = {}, {}
    vocab = BertTokenizer.from_pretrained(model_name).vocab
    tag = TagTokenizer().vocab
    shared = vocab.keys() & tag.keys()

    for s in shared:
        vocab2tag[int(vocab[s])] = int(tag[s])
        tag2vocab[int(tag[s])] = int(vocab[s])

    return vocab2tag, tag2vocab


class PACBertPreTrainedModel(nn.Module):
    """ PACBert pretrained model. """

    def __init__(self, config):
        """ Constructor for `PACBertPreTrainedModel`. 
        
        Args:
            config: Configuration for the PACBert model.
        """
        super(PACBertPreTrainedModel, self).__init__()
        self.config = config


    def _init_weights(self, module):
        """ Initialize model weights.
        
        Args:
            module: Module.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    @classmethod
    def from_pretrained(cls, model_name, config):
        """ Load Bert pretrained model.
        
        Args:
            model_name: Name of Bert pretrained model.
            config: Configuration for the PACBert model. 
        
        Returns:
            pacbert: PACBert model initialized from Bert pretrained model.
        """
        LOG.info(f"Loading pretrained model from {model_name}.")

        unshared_params = {"embeddings.token_type_embeddings.weight"}

        # Construct model.
        pacbert = cls(config)
        bert = BertModel.from_pretrained(model_name)

        # Get state_dict()
        pacbert_sd = pacbert.state_dict()
        bert_sd = bert.state_dict()

        # Get parameter names.
        # Remove prefix (if any).
        prefix = "pacbert." if list(pacbert_sd.keys())[0].startswith("pacbert.") else ""
        pacbert_params = [p.removeprefix(prefix) for p in pacbert_sd.keys()] if prefix else pacbert_sd.keys()
        bert_params = bert_sd.keys()

        # Get shared parameters.
        shared_params = (pacbert_params & bert_params) - unshared_params

        # Transfer pretrained weights.
        for name in shared_params:
            assert pacbert_sd[prefix + name].shape == bert_sd[name].shape, \
                f"Pretrained of shape {bert_sd[name].shape} does not match with shape {pacbert_sd[prefix + name].shape}."
            pacbert_sd[prefix + name] = bert_sd[name]
        
        # Initialize tag embedding with the same word embedding from Bert pretrained model.
        # Get mapping between the same word and tag.
        _, tag2vocab = vocab_tag_mapping(model_name)

        bert_word_emb_name = "embeddings.word_embeddings.weight"
        pacbert_tag_emb_name = prefix + "embeddings.bert_tag_embeddings.weight"

        bert_word_emb = bert_sd[bert_word_emb_name]
        pacbert_tag_emb = pacbert_sd[pacbert_tag_emb_name]

        pacbert_tag_emb[list(tag2vocab.keys())] = bert_word_emb[list(tag2vocab.values())]
        pacbert_sd[pacbert_tag_emb_name] = pacbert_tag_emb
        pacbert.load_state_dict(pacbert_sd)

        return pacbert


class PACBertEmbeddings(nn.Module):
    """ Construct the embeddings from user, text, tag, position, and token_type. """

    def __init__(self, config):
        """ Constructor for `PACBertEmbedding`.
        
        Args:
            config: Configuration for the PACBert model.
        """
        super(PACBertEmbeddings, self).__init__()
        self.user_embeddings = nn.Embedding(config.user_size, config.gnn_hidden_size)
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.bert_tag_embeddings = nn.Embedding(config.tag_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.gnn_tag_embeddings = nn.Embedding(config.tag_size, config.gnn_hidden_size, padding_idx=config.pad_token_id)

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.user_transform = nn.Linear(config.gnn_hidden_size, config.hidden_size)
        self.tag_transform = nn.Linear(config.gnn_hidden_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


    def forward(self, user_ids, text_ids, tag_ids):
        """ Process of `PACBertEmbeddings`.

        Args:
            user_ids : ID of user.           [batch_size, 1]
            text_ids : IDs of text sequence. [batch_size, max_txt_seq]
            tag_ids  : IDs of tag sequence.  [batch_size, max_tag_seq]

        Returns:
            embeddings     : Concatenated input embeddings. [batch_size, max_seq, hidden_size]
            attention_mask : Attention mask.                [batch_size, max_seq]
        """
        # Lookup user embedding and transform gnn subspace into bert subspace.
        user_embeddings = self.user_embeddings(user_ids) # [batch_size, 1, gnn_hidden_size]
        user_embeddings = self.user_transform(user_embeddings) # [batch_size, 1, hidden_size]

        # Lookup text embeddings.
        text_embeddings = self.word_embeddings(text_ids) # [batch_size, max_txt_seq, hidden_size]

        # Lookup gnn-based tag embeddings and transform into bert subspace.
        gnn_tag_embeddings = self.gnn_tag_embeddings(tag_ids) # [batch_size, max_tag_seq, gnn_hidden_size]
        gnn_tag_embeddings = self.tag_transform(gnn_tag_embeddings) # [batch_size, max_tag_seq, hidden_size]

        # Combine gnn-based and bert-based tag embeddings.
        bert_tag_embeddings = self.bert_tag_embeddings(tag_ids) # [batch_size, max_tag_seq, hidden_size]
        tag_embeddings = gnn_tag_embeddings * bert_tag_embeddings # [batch_size, max_tag_seq, hidden_size]

        cls_embeddings = text_embeddings[:, :1] # Get [CLS] embedding.
        text_embeddings = text_embeddings[:, 1:] # Remove [CLS] embedding.

        batch_size = text_embeddings.shape[0] 
        hidden_size = text_embeddings.shape[-1]

        text_mask = (text_ids > 0)[:, 1:] # Remove [CLS] token.
        tag_mask = tag_ids > 0

        # Get [SEP] embedding.
        sep_idx = (text_mask.sum(dim=-1, keepdim=True) - 1)
        sep_idx = sep_idx.unsqueeze(-1).repeat(1, 1, hidden_size)
        sep_embeddings = torch.gather(text_embeddings, 1, sep_idx)  # [batch_size, 1, hidden_size]

        # Prepare user, text, and tag indexes for concatenation.
        user_end = user_ids.new_zeros((batch_size, 1)) + 3 # Include [CLS], [user], and [SEP] tokens.
        text_end = user_end + text_mask.sum(1, keepdim=True)
        tag_end = text_end + tag_mask.sum(1, keepdim=True)
        max_length = (user_ids.shape[-1] + text_mask.sum(-1) + tag_mask.sum(-1)).max() + 3 # Include [CLS] and two [SEP] tokens.
        _, grid_pos = torch.meshgrid(torch.arange(batch_size, dtype=torch.long, device=text_embeddings.device),
                                    torch.arange(max_length, dtype=torch.long, device=text_embeddings.device),
                                    indexing='ij')
        embeddings = text_embeddings.new_zeros((batch_size, max_length, hidden_size))

        # Concatenate user, text, and tag embeddings.
        user_cat = torch.cat([cls_embeddings, user_embeddings, sep_embeddings], dim=1)
        embeddings[:, :user_cat.shape[1]] = user_cat
        embeddings[(grid_pos >= user_end) & (grid_pos < text_end)] = text_embeddings[text_mask]
        embeddings[(grid_pos >= text_end) & (grid_pos < tag_end)] = tag_embeddings[tag_mask]
        embeddings[grid_pos == tag_end] = sep_embeddings.squeeze()

        # Create token-type embeddings.
        token_type_ids = user_ids.new_zeros((batch_size, max_length)) # 0-user
        token_type_ids[(grid_pos >= user_end) & (grid_pos < text_end)] = 1 # 1-text
        token_type_ids[(grid_pos >= text_end) & (grid_pos <= tag_end)] = 2 # 2-tag
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # Create position embeddings.
        position_ids = grid_pos.clone()

        # Set the same position for all tags.
        tag_pos = text_end.repeat(1, max_length)
        tag_idx = (grid_pos >= text_end) & (grid_pos < tag_end)
        position_ids[tag_idx] = tag_pos[tag_idx]

        # Concatenate the rest position.
        end_pos = max_length - tag_mask.sum(-1, keepdim=True)
        temp_pos = grid_pos[(grid_pos > text_end) & (grid_pos <= end_pos)]
        position_ids[grid_pos >= tag_end] = temp_pos
        position_embeddings = self.position_embeddings(position_ids)

        # Create attention mask to avoid performing attention on padding tokens.
        attention_mask = text_mask.new_zeros((batch_size, max_length))
        attention_mask[grid_pos <= tag_end] = 1

        # Sum embeddings of input, position, and token_type together.
        embeddings = embeddings + token_type_embeddings + position_embeddings # [batch_size, max_length, hidden_size]
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings, attention_mask


class PACBertModel(PACBertPreTrainedModel):
    """ PACBert model. """

    def __init__(self, config, add_pooling_layer=False):
        """ Constructor for `PACBertModel`.

        Args:
            config: Configuration for the PACBert model.
            add_pooling_layer: Whether to use the pooling layer (Default=True).
        """
        super(PACBertModel, self).__init__(config)
        self.embeddings = PACBertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights.
        self.apply(self._init_weights)


    def forward(self,
                user_ids,
                text_ids,
                tag_ids,
                output_hidden_states=True,
                output_attentions=True):
        """ Process of `PACBertModel`.
        
        Args:
            user_ids : ID of user.           [batch_size, 1]
            text_ids : IDs of text sequence. [batch_size, max_txt_seq]
            tag_ids  : IDs of tag sequence.  [batch_size, max_tag_seq]
            output_hidden_states: Whether to output hidden states (Default=True).
            output_attentions: Whether to output attention weights (Default=True).

        Returns:
            sequence_output: Last layer hidden state of sequence tokens.
            pooled_output: Last layer hidden state of [CLS] token.
            encoder_output: Hidden states and attention weights (if output=True).
        """

        embedding_output, attention_mask = self.embeddings(user_ids=user_ids,
                                                           text_ids=text_ids,
                                                           tag_ids=tag_ids)
        
        extended_attention_mask = self._get_extended_attention_mask(attention_mask)

        encoder_outputs = self.encoder(hidden_states=embedding_output,
                                        attention_mask=extended_attention_mask,
                                        output_hidden_states=output_hidden_states,
                                        output_attentions=output_attentions)

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return (sequence_output, pooled_output) + encoder_outputs[1:]


    def _get_extended_attention_mask(self, attention_mask):
        """ Extend the attention mask before inputting into the encoder.
        
        Args:
            attention_mask: Attention mask.
        
        Returns:
            extended_attention_mask: Extended attention mask.
        """
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask


class PACBertRecommendationHead(nn.Module):
    """ Recommendation head for PACBert model. """

    def __init__(self, config):
        """ Constructor for `PACBertRecommendationHead`.
        
        Args:
            config: Configuration for the PACBert model.
        """
        super(PACBertRecommendationHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        cls_dropout = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        self.dropout = nn.Dropout(cls_dropout)
        self.cls = nn.Linear(config.hidden_size, config.tag_size, bias=False)
        self.cls.bias = nn.Parameter(torch.zeros(config.tag_size))


    def forward(self, hidden_states):
        """ Process of `PACBertRecommendationHead`.
        
        Args:
            hidden_states: Hidden states.
        
        Returns:
            hidden_states: Recommendation output.
        """
        hidden_states = self.transform(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.cls(hidden_states)
        return hidden_states


class PACBertForRecommendation(PACBertPreTrainedModel):
    """ PACBert model for recommendation. """
    
    def __init__(self, config):
        """ Constructor for `PACBertForRecommendation`
        
        Args:
            config: Configuration for the PACBert model.
        """
        super(PACBertForRecommendation, self).__init__(config)
        self.config = config
        self.pacbert = PACBertModel(self.config)
        self.rec = PACBertRecommendationHead(self.config)

        # Initialize weights.
        self.apply(self._init_weights)


    def forward(self, user_ids, text_ids, tag_ids):
        """ Process of `PACBertForRecommendation`.

        Args:
            user_ids : ID of user.           [batch_size, 1]
            text_ids : IDs of text sequence. [batch_size, max_txt_seq]
            tag_ids  : IDs of tag sequence.  [batch_size, max_tag_seq]

        Returns:
            outputs: Recommendation outputs that consist of:
            * logits: Recommendation outputs. [batch_size, max_seq, tag_size]
            * etc.: pooled_layer or hidden_states or attention_weight (if output=True)
        """
        outputs = self.pacbert(user_ids, text_ids, tag_ids)
        sequence_output = outputs[0]
        
        logits = self.rec(sequence_output)
        logits = logits[:, -(tag_ids.shape[-1]+1):-1] # Exclude the last [SEP].
        logits[:,:,[0,2]] = -10000.0 # Prevent model to predict [PAD] and [MASK] as output.

        outputs = (logits,) + outputs[2:]

        return outputs
import torch

from dataclasses import dataclass
from typing import Tuple


@dataclass
class AttentiveAggregationOutput:
    """ Outputs for class `AttentiveAggregation`.

    Args:
        attention_output: Attention output.     `[..., hidden_size]`
        attention_probs: Attention probability. `[num_attention_heads, ...]`
    """
    attention_output: torch.FloatTensor = None
    attention_probs: torch.FloatTensor = None


@dataclass
class RelationGNNOutput:
    """ Outputs for class `RelationGNN`.

    Args:
        src : Updated embedding of the source node. `[src_size, hidden_size]`
        tgt : Updated embedding of the target node. `[tgt_size, hidden_size]`
        src_attention : Attention scores of source messages toward the target node. 
            `[num_attention_heads, num_interactions]`
        tgt_attention : Attention scores of target messages toward the source node. 
            `[num_attention_heads, num_interactions]`
        src_attention_mask : Attention mask for source messages. `[src_size]`
        tgt_attention_mask : Attention mask for target messages. `[tgt_size]`
    """
    src: torch.FloatTensor = None
    tgt: torch.FloatTensor = None
    
    src_attention: Tuple[torch.FloatTensor] = None
    tgt_attention: Tuple[torch.FloatTensor] = None
    
    src_attention_mask: torch.FloatTensor = None
    tgt_attention_mask: torch.FloatTensor = None


@dataclass
class MANGNNLayerOutput:
    """ Outputs for class `MANGNNLayer`.

    Args:
        user: Updated user embedding. `[user_size, hidden_size]`
        tag: Updated tag embedding.   `[tag_size, hidden_size]`
        user_relation_attentions: Attention scores of each user's relation
            toward the user. `[num_attention_heads, 1, user_relation_size + 1]`
        tag_relation_attentions: Attention scores of each tag's relation
            toward the tag. `[num_attention_heads, 1, tag_relation_size + 1]`
        user_msg_attentions: Tuple of attention scores of each message in each relation
            toward the user. `(user_relation_size, [num_attention_heads, num_interactions])`
        tag_msg_attentions: Tuple of attention scores of each message in each relation
            toward the tag. `(tag_relation_size, [num_attention_heads, num_interactions])`
    """
    user: torch.FloatTensor = None
    tag: torch.FloatTensor = None

    user_relation_attentions: torch.FloatTensor = None
    tag_relation_attentions: torch.FloatTensor = None

    user_msg_attentions: Tuple[torch.FloatTensor] = None
    tag_msg_attentions: Tuple[torch.FloatTensor] = None


@dataclass
class MANGNNModelOutput:
    """ Outputs for class `MANGNNModel`.

    Args:
        last_user: User embedding from the last GNN layer.
            `[user_size, hidden_size]`
        last_tag: Tag embedding from the last GNN layer.
            `[tag_size, hidden_size]`
        users: Tuple of user embeddings from all GNN layers.
            `(num_layers, [user_size, hidden_size])`
        tags: Tuple of tag embeddings from all GNN layers.
            `(num_layers, [tag_size, hidden_size])`
        user_relation_attentions: Tuple of attention scores
            of each user's relation toward the user from all layers.
            `(num_layers, [num_attention_heads, 1, user_relation_size + 1])`
        tag_relation_attentions: Tuple of attention scores
            of each tag's relation toward the tag from all layers.
            `(num_layers, [num_attention_heads, 1, tag_relation_size + 1])`
        user_msg_attentions: Tuple of attention scores of each message
            in each relation toward the user from all layers.
            `(num_layers, user_relation_size, [num_attention_heads, num_interactions])`
        tag_msg_attentions: Tuple of attention scores of each message
            in each relation toward the tag from all layers.
            `(num_layers, tag_relation_size, [num_attention_heads, num_interactions])`
    """
    last_user: torch.FloatTensor = None
    last_tag: torch.FloatTensor = None

    users: Tuple[torch.FloatTensor] = None
    tags: Tuple[torch.FloatTensor] = None

    user_relation_attentions: Tuple[torch.FloatTensor] = None
    tag_relation_attentions: Tuple[torch.FloatTensor] = None

    user_msg_attentions: Tuple[Tuple[torch.FloatTensor]] = None
    tag_msg_attentions: Tuple[Tuple[torch.FloatTensor]] = None


@dataclass
class MANGNNForPredictionOutput(MANGNNModelOutput):
    """ Outputs for class `MANGNNForPrediction`.

    Args:
        logits: Prediction outputs. `[batch_size, num_labels]`
    """
    logits: torch.FloatTensor = None
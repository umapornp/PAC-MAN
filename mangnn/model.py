import math
import torch
import torch.nn as nn

from mangnn.utils.modeloutputs import (
    AttentiveAggregationOutput,
    RelationGNNOutput,
    MANGNNLayerOutput,
    MANGNNModelOutput,
    MANGNNForPredictionOutput
)


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu}


class PretrainedModel(nn.Module):
    """ Pretrained model. """
    
    def __init__(self, config):
        """ Constructor for `PretrainedModel`.

        Args:
            config: Configuration.
        """
        super(PretrainedModel, self).__init__()
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

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    
    @classmethod
    def from_pretrained(cls, model_path, config, graphs, device='cpu'):
        """ Load pretrained model. 

        Args:
            model_path: Path of pretrained model.
            config: Configuration. 
            graphs: List of `Graph` objects containing structure for GNNs.
            device: Running device. (Default=`cpu`).

        Returns:
            model: Pretrained MANGNN model.  
        """
        prefix = "mangnn."
        model = cls(config, graphs)

        # Load state_dict of model and pretrain.
        model_sd = model.state_dict()
        pretrain_sd = torch.load(model_path, map_location=device)
        pretrain_sd = pretrain_sd["MODEL"] if "MODEL" in pretrain_sd.keys() else pretrain_sd

        # Check whether model and pretrain have a prefix.
        has_prefix_model = list(model_sd.keys())[0].startswith(prefix)
        has_prefix_pretrain = list(pretrain_sd.keys())[0].startswith(prefix)

        # Get model parameters.
        model_params = model_sd.keys()

        # Transfer pretrained weights.
        for name in model_params:

            # Map parameter names between model and pretrain.
            if has_prefix_model and not has_prefix_pretrain:
                pretrain_name = name.removeprefix(prefix)
                shared_name = prefix + name
            elif not has_prefix_model and has_prefix_pretrain:
                pretrain_name = prefix + name
                shared_name = name.removeprefix(prefix)
            else:
                pretrain_name = name
                shared_name = name

            # Transfer pretrained weights.
            if shared_name in model_params:
                model_sd[shared_name] = pretrain_sd[pretrain_name]

        model.load_state_dict(model_sd)
        return model


class MessageAttention(nn.Module):
    """ Aggregate messages based on attention. """
    
    def __init__(self, config):
        """ Constructor for `MessageAttention`.

        Args:
            config: Configuration.
        """
        super(MessageAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)


    def transpose_for_scores(self, x):
        """ Transpose tensor before performing attention.

        Args:
            x: Input tensor. `[..., hidden_size]`

        Returns:
            x: Transposed tensor. `[num_attention_heads, ..., attention_head_size]`
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        x = x.permute(1, 0, 2) if self.is_sparse else x.permute(0, 2, 1, 3)
        return x


    def merge_head(self, x):
        """ Merge attention heads.

        Args:
            x: Tensor of all attention heads. `[num_attention_heads, ..., attention_head_size]`

        Returns:
            x: Merged tensor. `[..., hidden_size]`
        """
        x = x.permute(1, 0, 2) if self.is_sparse else x.permute(0, 2, 1, 3)
        new_x_shape = x.size()[:-2] + (self.all_head_size,)
        x = x.contiguous().view(*new_x_shape)
        return x


    def segment_softmax(self, input, dim, index):
        """ Compute softmax over index.

        Args:
            input : Input tensor.
            dim   : Dimension of index.
            index : Indices to perform softmax function.

        Returns:
            softmax: Softmax tensor.
        """
        exp = torch.exp(input)
        exp_sum = exp.new_zeros(exp.shape)
        exp_sum = exp_sum.index_add(dim=dim, index=index, source=exp).index_select(dim=-1, index=index)  
        softmax = exp / exp_sum
        return softmax
    
    
    def scaled_dot_product_sparse(self, query_layer, key_layer, value_layer, q_indices):
        """ Perform a scaled dot product operation on the sparse tensor.

        Args:
            query_layer : Query. `[num_attention_heads, ..., attention_head_size]`
            key_layer   : Key.   `[num_attention_heads, ..., attention_head_size]`
            value_layer : Value. `[num_attention_heads, ..., attention_head_size]`
            q_indices   : Indices of query along key and value.

        Returns:
            context_layer   : Attention output.      `[num_attention_heads, ..., attention_head_size]`
            attention_probs : Attention probability. `[num_attention_heads, ...]`
        """
        attention_scores = torch.sum(query_layer * key_layer, dim=-1)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = self.segment_softmax(attention_scores, dim=-1, index=q_indices)
        attention_probs = self.dropout(attention_probs).unsqueeze(-1)

        context_layer = value_layer.new_zeros(value_layer.shape)
        context_layer = context_layer.index_add(dim=1, index=q_indices, source=attention_probs * value_layer)
        context_layer = context_layer.index_select(index=q_indices.unique(), dim=1)
        
        return context_layer, attention_probs
    
    
    def scaled_dot_product(self, query_layer, key_layer, value_layer, attention_mask):
        """ Perform a scaled dot product operation. 

        Args:
            query_layer     : Query. `[num_attention_heads, ..., attention_head_size]`
            key_layer       : Key.   `[num_attention_heads, ..., attention_head_size]`
            value_layer     : Value. `[num_attention_heads, ..., attention_head_size]`
            attention_mask  : Attention mask. [num_attention_heads, ...]`

        Returns:
            context_layer   : Attention output.      `[num_attention_heads, ..., attention_head_size]`
            attention_probs : Attention probability. `[num_attention_heads, ...]`
        """
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        
        return context_layer, attention_probs


    def forward(
        self,
        q,
        k,
        v,
        attention_mask=None,
        output_attentions=False,
        is_sparse=False,
        q_indices=None,
        k_indices=None
    ):
        """ Process of `MessageAttention`. 

        Args:
            q : Query. `[..., hidden_size]`
            k : Key.   `[..., hidden_size]`
            v : Value. `[..., hidden_size]`
            attention_mask : Attention mask. `[num_attention_heads, ...]`
            output_attentions : Whether to output attention scores (Default=`False`).
            is_sparse : Whether the tensor is sparse (Default=`False`).
            q_indices : Indices of query.
            k_indices : Indices of key.

        Returns:
            Tuple of:
            * context_layer   : Aggregated messages.   `[..., hidden_size]`
            * attention_probs : Attention probability. `[num_attention_heads, ...]`
        """
        self.is_sparse = is_sparse
        
        if self.is_sparse:
            q = q.index_select(index=q_indices, dim=0) # [num_interactions, hidden_size]
            k = k.index_select(index=k_indices, dim=0) # [num_interactions, hidden_size] 
            v = v.index_select(index=k_indices, dim=0) # [num_interactions, hidden_size]

        query_layer = self.transpose_for_scores(self.query(q)) # [num_attention_heads, ..., attention_head_size]
        key_layer = self.transpose_for_scores(self.key(k))     # [num_attention_heads, ..., attention_head_size]
        value_layer = self.transpose_for_scores(self.value(v)) # [num_attention_heads, ..., attention_head_size]
        
        if self.is_sparse:
            context_layer, attention_probs = self.scaled_dot_product_sparse(
                query_layer=query_layer,
                key_layer=key_layer,
                value_layer=value_layer,
                q_indices=q_indices
            )
        else:
            context_layer, attention_probs = self.scaled_dot_product(
                query_layer=query_layer,
                key_layer=key_layer,
                value_layer=value_layer,
                attention_mask=attention_mask
            )

        context_layer = self.merge_head(context_layer) # [..., hidden_size]
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class MessageUpdate(nn.Module):
    """ Update aggregated messages. """
    
    def __init__(self, config):
        """ Constructor for `MessageUpdate`.

        Args:
            config: Configuration.
        """
        super(MessageUpdate, self).__init__()
        self.activation = ACT2FN[config.activation]
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        

    def forward(self, hidden_states, input_tensor):
        """ Process of `MessageUpdate`.

        Args:
            hidden_states: Aggregated messages from the current layer. `[..., hidden_size]`
            input_tensor: Messages from the previous layer. `[..., hidden_size]`

        Returns:
            hidden_states: Updated messages. `[..., hidden_size]`
        """
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class AttentiveAggregation(nn.Module):
    """ Attentive aggregation: aggregate and update messages based on attention. """
    
    def __init__(self, config):
        """ Constructor for `AttentiveAggregation`.

        Args:
            config: Configuration.
        """
        super(AttentiveAggregation, self).__init__()
        self.attention = MessageAttention(config)
        self.update = MessageUpdate(config)


    def forward(
        self,
        q,
        k,
        v,
        attention_mask=None,
        output_attentions=False,
        is_sparse=False,
        q_indices=None,
        k_indices=None
    ):
        """ Process of `AttentiveAggregation`.

        Args:
            q: Query. `[..., hidden_size]`
            k: Key.   `[..., hidden_size]`
            v: Value. `[..., hidden_size]`
            attention_mask: Attention mask. [num_attention_heads, ...]
            output_attentions: Whether to output attention scores (Default=`False`).
            is_sparse: Whether the tensor is sparse (Default=`False`). 
            q_indices: Indices of query.
            k_indices: Indices of key.

        Returns:
            attention_output: Attention output.      `[..., hidden_size]`
            attention_probs:  Attention probability. `[num_attention_heads, ...]`
        """
        # Aggregate messages.
        self_outputs = self.attention(
            q, k, v,
            attention_mask,
            output_attentions,
            is_sparse,
            q_indices,
            k_indices
        )

        # Lookup unique queries from sparse indices if the tensor is sparse.
        q = q.index_select(index=q_indices.unique(), dim=0) if is_sparse else q 

        # Update messages.
        attention_output = self.update(self_outputs[0], q)

        return AttentiveAggregationOutput(
            attention_output=attention_output,
            attention_probs=self_outputs[1:] # Return atttention prob. if output=True.
        )


class RelationGNN(nn.Module):
    """ Relation GNN. """
    
    def __init__(self, config, graph):
        """ Constructor for `RelationGNN`.

        Args:
            config : Configuration.
            graph  : `Graph` object containing structure for GNN.
        """
        super(RelationGNN, self).__init__()
        self.name = graph.name
        self.adj = graph.adj
        self.src_size, self.tgt_size = graph.adj.size()
        self.src_ids, self.tgt_ids = graph.adj._indices() # [num_interactions]
        self.src_type, self.tgt_type = graph.src_type, graph.tgt_type
        
        self.agg_src2tgt = AttentiveAggregation(config) if graph.agg_src else None
        self.agg_tgt2src = AttentiveAggregation(config) if graph.agg_tgt else None


    def forward(self, src_emb, tgt_emb, output_attentions=False):
        """ Process of `RelationGNN`.

        Args:
            src_emb : Embedding of the source node. `[src_size, hidden_size]`
            tgt_emb : Embedding of the target node. `[tgt_size, hidden_size]`
            output_attentions : Whether to output attention scores (Default=`False`).

        Returns:
            src : Updated embedding of the source node. `[src_size, hidden_size]`
            tgt : Updated embedding of the target node. `[tgt_size, hidden_size]`
            src_attention : Attention scores of source messages toward the target node. 
                `[num_attention_heads, num_interactions]`
            tgt_attention : Attention scores of target messages toward the source node. 
                `[num_attention_heads, num_interactions]`
            src_attention_mask : Attention mask for source messages. `[src_size]`
            tgt_attention_mask : Attention mask for target messages. `[tgt_size]`
        """
        new_src_emb, new_tgt_emb, src_attention, tgt_attention, src_attention_mask, tgt_attention_mask = (None,)*6

        # Aggregate source node >>> targat node.
        if self.agg_src2tgt:
            
            # Aggregate and update messages
            tgt = self.agg_src2tgt(
                q=tgt_emb, # [tgt_size, hidden_size]
                k=src_emb, # [src_size, hidden_size]
                v=src_emb, # [src_size, hidden_size]
                output_attentions=output_attentions,
                is_sparse=True,
                q_indices=self.tgt_ids, # [num_interactions]
                k_indices=self.src_ids  # [num_interactions]
            )

            # Update new embedding to the target node.
            new_tgt_emb = tgt_emb.new_zeros(tgt_emb.shape)
            new_tgt_emb[self.tgt_ids.unique()] = tgt.attention_output

            # Output attention scores.
            tgt_attention = tgt.attention_probs

            # Create an attention mask for each node. (0-no interactions, 1-have interactions)
            tgt_attention_mask = self.tgt_ids.new_zeros(self.tgt_size)
            tgt_attention_mask[self.tgt_ids.unique()] = 1

        # Aggregate target node >>> source node.
        if self.agg_tgt2src:
            
            # Aggregate and update messages.
            src = self.agg_tgt2src(
                q=src_emb, # [src_size, hidden_size]
                k=tgt_emb, # [tgt_size, hidden_size]
                v=tgt_emb, # [tgt_size, hidden_size]
                output_attentions=output_attentions,
                is_sparse=True,
                q_indices=self.src_ids, # [num_interactions]
                k_indices=self.tgt_ids  # [num_interactions]
            )
            
            # Update new embedding to the source node.
            new_src_emb = src_emb.new_zeros(src_emb.shape)
            new_src_emb[self.src_ids.unique()] = src.attention_output

            # Output attention scores.
            src_attention = src.attention_probs

            # Create an attention mask for each node. (0-no interactions, 1-have interactions)
            src_attention_mask = self.src_ids.new_zeros(self.src_size)
            src_attention_mask[self.src_ids.unique()] = 1

        return RelationGNNOutput(
            src=new_src_emb,
            tgt=new_tgt_emb,
            src_attention=src_attention,
            tgt_attention=tgt_attention,
            src_attention_mask=src_attention_mask,
            tgt_attention_mask=tgt_attention_mask
        )


class MANGNNLayer(nn.Module):
    """ MANGNN layer. """
    
    def __init__(self, config, graphs):
        """ Constructor for `MANGNNLayer`.

        Args:
            config: Configuration.
            graphs: List of `Graph` objects containing structure for GNNs.
        """
        super(MANGNNLayer, self).__init__()
        self.relation_gnns = nn.ModuleList([RelationGNN(config, graph) for graph in graphs])
        self.agg_user = AttentiveAggregation(config)
        self.agg_tag = AttentiveAggregation(config)


    def forward(self, user_emb, tag_emb, output_attentions=False):
        """" Process of `MANGNNLayer`.

        Args:
            user_emb: User embedding. `[user_size, hidden_size]`
            tag_emb: Tag embedding.   `[tag_size, hidden_size]`
            output_attentions: Whether to output attention scores (Default=`False`).

        Returns:
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
        emb = {"user": user_emb, "tag": tag_emb}
        all_relations = {"user": (), "tag": ()}
        all_attentions = {"user": (), "tag": ()}
        all_attention_mask = {"user": (), "tag": ()}

        # Loop each relation GNN.
        for relation_gnn in self.relation_gnns:

            # Aggregate and update messages of each relation.
            gnn_outputs = relation_gnn(
                emb[relation_gnn.src_type],
                emb[relation_gnn.tgt_type],
                output_attentions
            )

            if gnn_outputs.src is not None:
                all_relations[relation_gnn.src_type] += (gnn_outputs.src,)
                all_attention_mask[relation_gnn.src_type] += (gnn_outputs.src_attention_mask,)

                if output_attentions:
                    all_attentions[relation_gnn.src_type] += (gnn_outputs.src_attention,)

            if gnn_outputs.tgt is not None:
                all_relations[relation_gnn.tgt_type] += (gnn_outputs.tgt,)
                all_attention_mask[relation_gnn.tgt_type] += (gnn_outputs.tgt_attention_mask,)

                if output_attentions:
                    all_attentions[relation_gnn.tgt_type] += (gnn_outputs.tgt_attention,)

        # Stack messages of all relations and the self-node.
        all_user_relations = torch.stack(all_relations["user"] + (user_emb,), dim=1)
        all_tag_relations = torch.stack(all_relations["tag"] + (tag_emb,), dim=1)

        # Create attention mask for the self-node.
        user_self_mask = all_attention_mask["user"][0].new_ones(all_attention_mask["user"][0].shape)
        tag_self_mask = all_attention_mask["tag"][0].new_ones(all_attention_mask["tag"][0].shape)

        # Stack attention mask of all relations and the self-node.
        all_user_attention_mask = torch.stack(all_attention_mask["user"] + (user_self_mask,), dim=1)
        all_tag_attention_mask = torch.stack(all_attention_mask["tag"] + (tag_self_mask,), dim=1)

        # Extend attention mask.
        all_user_attention_mask = self._get_extended_attention_mask(all_user_attention_mask)
        all_tag_attention_mask = self._get_extended_attention_mask(all_tag_attention_mask)

        # Aggregate all user relations.
        new_user_emb = self.agg_user(
            q=user_emb.unsqueeze(1),  # [user_size, 1, hidden_size]
            k=all_user_relations,     # [user_size, user_relation_size + 1, hidden_size]
            v=all_user_relations,     # [user_size, user_relation_size + 1, hidden_size]
            attention_mask=all_user_attention_mask, # [user_size, user_relation_size + 1]
            output_attentions=output_attentions,
        )

        # Aggregate all tag relations.
        new_tag_emb = self.agg_tag(
            q=tag_emb.unsqueeze(1),  # [tag_size, 1, hidden_size]
            k=all_tag_relations,     # [tag_size, tag_relation_size + 1, hidden_size]
            v=all_tag_relations,     # [tag_size, tag_relation_size + 1, hidden_size]
            attention_mask=all_tag_attention_mask, # [tag_size, tag_relation_size + 1]
            output_attentions=output_attentions,
        )

        return MANGNNLayerOutput(
            user=new_user_emb.attention_output.squeeze(1),
            tag=new_tag_emb.attention_output.squeeze(1),
            user_relation_attentions=new_user_emb.attention_probs,
            tag_relation_attentions=new_tag_emb.attention_probs,
            user_msg_attentions=all_attentions["user"],
            tag_msg_attentions=all_attentions["tag"],
        )
    
    
    def _get_extended_attention_mask(self, attention_mask):
        """ Extend the attention mask before inputting into the encoder.
        
        Args:
            attention_mask: Attention mask.
        
        Returns:
            extended_attention_mask: Extended attention mask.
        """
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


class MANGNNModel(PretrainedModel):
    """ MANGNN model. """
    
    def __init__(self, config, graphs):
        """ Constructor for `MANGNNModel`.

        Args:
            config: Configuration.
            graphs: List of `Graph` objects containing structure for GNNs.
        """
        super(MANGNNModel, self).__init__(config)
        self.user_embeddings = nn.Embedding(config.user_size, config.hidden_size)
        self.tag_embeddings = nn.Embedding(config.tag_size, config.hidden_size)
        self.layers = nn.ModuleList([MANGNNLayer(config, graphs) for _ in range(config.num_layers)])

        # Initialize weights.
        self.apply(self._init_weights)


    def forward(self, output_attentions=False):
        """ Process of `MANGNNModel`.

        Args:
            output_attentions: Whether to output attention scores (Default:`False`).

        Returns:
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
        # Embedding of user and tag.
        user_embeddings = self.user_embeddings.weight
        tag_embeddings = self.tag_embeddings.weight

        all_embeddings = {"user": (), "tag": ()}
        all_relation_attentions = {"user": (), "tag": ()}
        all_msg_attentions = {"user": (), "tag": ()}

        # Loop GNN layers.
        for layer in self.layers:
            
            # Gather embeddings from each layer.
            all_embeddings["user"] += (user_embeddings,)
            all_embeddings["tag"] += (tag_embeddings,)

            # GNN layer.
            layer_outputs = layer(user_embeddings, tag_embeddings, output_attentions)

            # Update embeddings of user and tag for the next layer.
            user_embeddings, tag_embeddings = layer_outputs.user, layer_outputs.tag

            # Gather attention scores from each layer.
            all_relation_attentions["user"] += layer_outputs.user_relation_attentions
            all_relation_attentions["tag"] += layer_outputs.tag_relation_attentions
            all_msg_attentions["user"] += layer_outputs.user_msg_attentions
            all_msg_attentions["tag"] += layer_outputs.tag_msg_attentions

        all_embeddings["user"] += (user_embeddings,)
        all_embeddings["tag"] += (tag_embeddings,)

        return MANGNNModelOutput(
            last_user=user_embeddings,
            last_tag=tag_embeddings,
            users=all_embeddings["user"],
            tags=all_embeddings["tag"],
            user_relation_attentions=all_relation_attentions["user"],
            tag_relation_attentions=all_relation_attentions["tag"],
            user_msg_attentions=all_msg_attentions["user"],
            tag_msg_attentions=all_msg_attentions["tag"],
        )


class MANGNNForPrediction(PretrainedModel):
    """ MANGNN model for prediction. """

    def __init__(self, config, graphs):
        """ Constructor for `MANGNNForPrediction`.

        Args:
            config: Configuration.
            graphs: List of `Graph` objects containing structure for GNNs.
        """
        super(MANGNNForPrediction, self).__init__(config)
        self.mangnn = MANGNNModel(config, graphs)
        classifier_dropout = (config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob)
        self.dropout = nn.Dropout(classifier_dropout)
        self.cls = nn.Linear(config.hidden_size * (config.num_layers + 1), config.num_labels)

        # Initialize weights.
        self.apply(self._init_weights)


    def forward(self, user_ids, tag_ids, output_attentions=False):
        """ Process of `MANGNNForPrediction`.

        Args:
            user_ids: ID of user.   `[batch_size]`
            tag_ids: ID of tag.     `[batch_size]`
            output_attentions: Whether to output attention scores (Default=`False`).

        Returns:
            logits: Prediction output.  `[batch_size, num_labels]`
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
        # MANGNN model.
        mangnn_outputs = self.mangnn(output_attentions)

        # Concatenate users/tag embeddings from all gnn layers along hidden_size dimension.
        user_embeddings = torch.cat(mangnn_outputs.users, dim=-1)
        tag_embeddings = torch.cat(mangnn_outputs.tags, dim=-1)

        # Lookup user and tag embeddings.
        user_emb = user_embeddings.index_select(index=user_ids, dim=0)
        tag_emb = tag_embeddings.index_select(index=tag_ids, dim=0)

        # Compute user profile.
        profile = user_emb * tag_emb

        # Classify labels.
        logits = self.cls(self.dropout(profile))

        return MANGNNForPredictionOutput(
            logits=logits,
            last_user=mangnn_outputs.last_user,
            last_tag=mangnn_outputs.last_tag,
            users=mangnn_outputs.users,
            tags=mangnn_outputs.tags,
            user_relation_attentions=mangnn_outputs.user_relation_attentions,
            tag_relation_attentions=mangnn_outputs.tag_relation_attentions,
            user_msg_attentions=mangnn_outputs.user_msg_attentions,
            tag_msg_attentions=mangnn_outputs.tag_msg_attentions,
        )
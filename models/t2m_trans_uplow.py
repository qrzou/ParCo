import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical

import models.pos_encoding as pos_encoding



class FuseModuleV1_3(nn.Module):
    """
    1_3 median trunck
    FuseModule Version 1: MLP reduce the dim at first layer
    Input is the concatenated feature from all the other parts, making the in_features is large.
    MLP using this large number as in and out feature dim, making itself has large parameters.
    """
    def __init__(self, in_features, out_features, num_mlp_layers=3, drop_out_rate=0.1, alpha=1.0):
        super().__init__()
        assert num_mlp_layers >= 2

        module_list = []

        # First layer
        # get the median dim for 1_3 version
        hidden_dim = (in_features + out_features) // 2
        module_list += [
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
        ]

        for i in range(num_mlp_layers - 1):

            in_dim = hidden_dim if i == 0 else out_features

            if i == num_mlp_layers - 1:  # Last layer
                module_list += [
                    nn.Linear(in_dim, out_features),
                    nn.Dropout(drop_out_rate)
                ]

            else:
                module_list += [
                    nn.Linear(in_dim, out_features),
                    nn.GELU()
                ]

        self.mlp = nn.Sequential(*module_list)
        self.alpha = alpha


    def forward(self, x, y):
        """
        out = LayerNorm( x + alpha * MLP(y) )
        out = x + alpha * MLP(y)
        :param x: (B, nframes, out_features)  local part token feature
        :param y: List containing other parts token feature
                  [(B,nframes,part0_dim),..., (B,nframes,part5_dim)],
                    there is a number between 0~5 which does not exist in the indexes.
                  After torch.cat it will be (B, nframes, in_features)
        :return:  (B, nframes, out_features)  fused feature
        """
        assert isinstance(y, list)
        assert len(y) == 1
        y = torch.cat(y, dim=2)  # (B, 51, other_parts_embed_dim)
        out = x + self.alpha * self.mlp(y)
        return out



class TransformerFuseHiddenDim(nn.Module):

    def __init__(self,
                 clip_dim=512,
                 block_size=16,     # T2M-GPT default: 51. same to the output motion sequence length
                 num_layers=18,     # remember our bodypart num_layer is aligned with the total block num, but T2M-GPT block num is 2*num_layer
                 n_head=8,
                 drop_out_rate=0.1,
                 fc_rate=4,

                 # FusionModule
                 use_fuse=True,     # use fused info from other parts?
                 fuse_ver='V1_2',
                 alpha=1.0,         # weight of global info
                 # Common transformer config
                 parts_code_nb={},    # size of part codebooks
                 parts_embed_dim={},  # dimension (size) of transformer attention block
                 # Fuse V1 config
                 num_mlp_layers=3,    #
                 # Fuse V2 config
                 fusev2_sub_mlp_out_features={},
                 fusev2_sub_mlp_num_layers=2,
                 fusev2_head_mlp_num_layers=2,

                 ):
        super().__init__()

        self.parts_name = ['Upper_body', 'Lower_body']
        self.num_layers = num_layers

        self.use_fuse = use_fuse
        self.fuse_ver = fuse_ver
        self.alpha = alpha
        # Common transformer config
        self.parts_code_nb = parts_code_nb
        self.parts_embed_dim = parts_embed_dim
        # Fuse V1 config
        self.num_mlp_layers = num_mlp_layers
        # Fuse V2 config
        self.fusev2_sub_mlp_out_features = fusev2_sub_mlp_out_features
        self.fusev2_sub_mlp_num_layers = fusev2_sub_mlp_num_layers
        self.fusev2_head_mlp_num_layers = fusev2_head_mlp_num_layers


        sum_parts_emb_dim = 0
        for name in self.parts_name:
            sum_parts_emb_dim += parts_embed_dim[name]

        for name in self.parts_name:
            # [Base]
            num_vq = parts_code_nb[name]
            embed_dim = parts_embed_dim[name]
            other_parts_embed_dim = sum_parts_emb_dim - embed_dim

            # FuseV2 params:
            sub_mlp_in_features = []
            sub_mlp_out_features = []
            for name2 in self.parts_name:
                if name2 != name:
                    sub_mlp_in_features.append(parts_embed_dim[name2])
                    sub_mlp_out_features.append(fusev2_sub_mlp_out_features[name2])


            tok_emb = nn.Embedding(num_vq + 3, embed_dim)  # 3: end token, pad token, mask token
            cond_emb = nn.Linear(clip_dim, embed_dim)


            pos_embed = pos_encoding.PositionEmbedding(block_size, embed_dim, 0.0, False)

            setattr(self, f'{name}_tok_emb', tok_emb)
            setattr(self, f'{name}_cond_emb', cond_emb)
            setattr(self, f'{name}_pos_embed', pos_embed)

            # [Transformer and Fuse block]
            for i in range(num_layers):
                # block
                block = Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate)
                setattr(self, f'{name}_block_{i}', block)

                # Fuse module for each layers except for the first transformer layer.
                if use_fuse and i != 0:
                    # LayerNorm before sent into fuse modules.
                    ln = nn.LayerNorm(embed_dim)
                    setattr(self, f'{name}_ln_{i}', ln)
                    # Fuse module: MLP, global info weight
                    # todo: set in the config file
                    if self.fuse_ver == 'V1_3':
                        fuse = FuseModuleV1_3(in_features=other_parts_embed_dim, out_features=embed_dim,
                                          num_mlp_layers=num_mlp_layers, drop_out_rate=drop_out_rate, alpha=alpha)
                    else:
                        raise NotImplementedError()
                    setattr(self, f'{name}_fuse_{i}', fuse)

            # [Head]
            '''blocks in head were moved into the [Base]'''
            ln_f = nn.LayerNorm(embed_dim)
            head = nn.Linear(embed_dim, num_vq + 1, bias=False)
            setattr(self, f'{name}_ln_f', ln_f)
            setattr(self, f'{name}_head', head)


        self.block_size = block_size
        self.apply(self._init_weights)


    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, parts_idxs, clip_feature):
        """
        Inputs: lists of [Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm]'s idx and clip features
        :param parts_idxs:          List: [(B, 50), ..., (B, 50)]
        :param clip_feature:  Tensor: (B, 512)
        :return: Each parts' predicted tokenized motion sequence logits.
        """

        # [Prepare token embeddings]
        parts_token_embeddings = []
        for i, name in enumerate(self.parts_name):

            # fetch input and module
            idx = parts_idxs[i]
            cond_emb = getattr(self, f'{name}_cond_emb')
            pos_embed = getattr(self, f'{name}_pos_embed')

            # Debug, make sure all elems have same length
            assert len(idx) == len(parts_idxs[0])

            if len(idx) == 0:
                '''
                Used by self.sample(), in the inference stage.
                The parts_idxs[name] is an empty list now. See self.sample() to check the input's details.
                '''
                token_embeddings = cond_emb(clip_feature).unsqueeze(1)  # (B, 512) => (B, 1, embed_dim)

            else:

                B, t = idx.size()  # t: token length
                assert t <= self.block_size, "Cannot forward, model block size is exhausted."

                # embed the tokens
                tok_emb = getattr(self, f'{name}_tok_emb')
                token_embeddings = tok_emb(idx)  # (B, 50) => (B, 50, embed_dim)

                # add text condition embed, get (B, 51, embed_dim)
                token_embeddings = torch.cat([cond_emb(clip_feature).unsqueeze(1), token_embeddings], dim=1)

            x = pos_embed(token_embeddings)  # (B, 51, embed_dim); add the position embeddings
            parts_token_embeddings.append(x)


        # [Go through the blocks]
        for i in range(self.num_layers):

            # Update the parts_token_embeddings using Transformer and Fuse

            # Create a list to keep the tensor from parts_token_embeddings,
            #   because the parts_token_embeddings will be updated.
            if self.use_fuse and i != 0:
                '''
                Do layerNorm before input to the FusionModule
                  It is unlikely to set the layerNorm into the FusionModule,
                    because the output will be sent to different FusionModule,
                    and we need to do LayerNorm before sending.
                  And insert a LayerNorm in the end in transformer block need to modify a lot of codes,
                    so we don't do that.
                '''
                no_modified_input = []
                for j, name in enumerate(self.parts_name):
                    ln = getattr(self, f'{name}_ln_{i}')
                    no_modified_input.append(ln(parts_token_embeddings[j]))
            else:
                no_modified_input = [elem for elem in parts_token_embeddings]  # not being modified parts token embeddings

            for j, name in enumerate(self.parts_name):

                # Fetch the input data and module
                block = getattr(self, f'{name}_block_{i}')
                x = no_modified_input[j]

                # Fuse the information or not.
                if self.use_fuse and i != 0:  # fuse the parts info if not 0-th layer

                    fuse = getattr(self, f'{name}_fuse_{i}')
                    other_parts_emb = [no_modified_input[count]
                                       for count in range(len(self.parts_name)) if count != j]
                    # other_parts_emb = torch.cat(other_parts_emb, dim=2)  # (B, 51, other_parts_embed_dim)
                    x = fuse(x, other_parts_emb)

                # send to transformer block
                x = block(x)

                # update parts_token_embeddings
                parts_token_embeddings[j] = x

        # [Heads]
        parts_logits = []
        for i, name in enumerate(self.parts_name):

            # Fetch the input data and module
            x = parts_token_embeddings[i]  # (B, 51, embed_dim)
            ln_f = getattr(self, f'{name}_ln_f')
            head = getattr(self, f'{name}_head')

            x = ln_f(x)       # (B, 51, embed_dim)
            logits = head(x)  # (B, 51, 513)  513: num_vq + 1 (End)

            parts_logits.append(logits)

        return parts_logits


    def sample(self, clip_feature, if_categorial=False):
        """
        This function is used in evaluation, to get the predicted motion code sequence.
        Only support 1 sample.
        Does not support Batch operation (clip_feature is a batch Tensor (B, 512)).

        clip_feature: Tensor (1, 512). Single clip feature.
        """

        # Assertion
        assert len(clip_feature.shape) == 2
        assert clip_feature.shape[0] == 1

        stop_generate = False
        min_part_seq_len = 0
        xs = [[] for _ in self.parts_name]  # create empty list

        for k in range(self.block_size):

            if stop_generate:
                break

            if k == 0:
                x = [[] for _ in self.parts_name]  # create empty list
            else:
                x = xs

            parts_logits = self.forward(x, clip_feature)  # List: [(B, seq_len, 513), ..., (B, seq_len, 513)]


            for i, name in enumerate(self.parts_name):

                logits = parts_logits[i]  # (B, seq_len, 513)
                out_seq_len = logits.shape[1]

                logits = logits[:, -1, :]  # get the last predicted token, (B, 513)
                probs = F.softmax(logits, dim=-1)

                if if_categorial:
                    dist = Categorical(probs)
                    idx = dist.sample()
                    if idx == self.parts_code_nb[name]:
                        stop_generate = True
                        min_part_seq_len = out_seq_len - 1
                        break  # stop generation if predict End token
                    idx = idx.unsqueeze(-1)

                else:
                    _, idx = torch.topk(probs, k=1, dim=-1)  # (B, 1)
                    if idx[0] == self.parts_code_nb[name]:  # idx[0] use the first sample in the batch, thus this function only support B == 1
                        stop_generate = True
                        min_part_seq_len = out_seq_len - 1
                        break  # stop generation if predict End token

                # append to the sequence and continue
                if k == 0:
                    xs[i] = idx
                    # xs = idx
                else:
                    xs[i] = torch.cat((xs[i], idx), dim=1)  # (B, n+1), B == 1
                    # xs = torch.cat((xs, idx), dim=1)  # (B, n+1), B == 1

            # return the result if reaches the max length
            if k == self.block_size - 1:
                for i, name in enumerate(self.parts_name):
                    xs[i] = xs[i][:, :-1]
                return xs

        # Unify all parts seq length to the min_part_seq_len
        if min_part_seq_len == 0:
            xs = [[] for _ in self.parts_name]
        else:
            for i in range(len(xs)):
                xs[i] = xs[i][:, :min_part_seq_len]

        return xs  # return the result if predict the End token


    def sample_batch(self, clip_feature, if_categorial=False):
        """
        This function is used in evaluation, to get the predicted motion code sequence.
        Support batch samples.
        clip_feature: Tensor (B, 512). Single clip feature.
        """

        # Assertion
        assert len(clip_feature.shape) == 2

        xs = [[] for _ in self.parts_name]  # create empty list

        for k in range(self.block_size):

            if k == 0:
                x = [[] for _ in self.parts_name]  # create empty list
            else:
                x = xs

            parts_logits = self.forward(x, clip_feature)  # List: [(B, seq_len, 513), ..., (B, seq_len, 513)]

            for i, name in enumerate(self.parts_name):

                logits = parts_logits[i]  # (B, seq_len, 513)
                out_seq_len = logits.shape[1]

                logits = logits[:, -1, :]  # get the last predicted token, (B, 513)
                probs = F.softmax(logits, dim=-1)

                if if_categorial:
                    dist = Categorical(probs)
                    idx = dist.sample()
                    idx = idx.unsqueeze(-1)

                else:
                    _, idx = torch.topk(probs, k=1, dim=-1)  # (B, 1)

                # append to the sequence and continue
                if k == 0:
                    xs[i] = idx

                else:
                    xs[i] = torch.cat((xs[i], idx), dim=1)  # (B, n+1), B == 1

        # Drop the last End token
        for i, name in enumerate(self.parts_name):
            xs[i] = xs[i][:, :-1]

        return xs  # List: [(B, 50), ..., (B, 50)]



class CausalCrossConditionalSelfAttention(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # The correct code reflecting the author's purpose may be following:
        #  Because in the forward function, the embeddings will be split by n_head.
        assert embed_dim % n_head == 0

        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        '''
        Causal mask to ensure that attention is only applied to the left in the input sequence
        Example: when block_size == 8
        [[[[1., 0., 0., 0., 0., 0., 0., 0.],
           [1., 1., 0., 0., 0., 0., 0., 0.],
           [1., 1., 1., 0., 0., 0., 0., 0.],
           [1., 1., 1., 1., 0., 0., 0., 0.],
           [1., 1., 1., 1., 1., 0., 0., 0.],
           [1., 1., 1., 1., 1., 1., 0., 0.],
           [1., 1., 1., 1., 1., 1., 1., 0.],
           [1., 1., 1., 1., 1., 1., 1., 1.]]]]
        '''
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
        self.n_head = n_head

    def forward(self, x):
        # B: batch size
        # T: token sequence length
        # C: embed_dim
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))  # fill '-inf' to the masked attention
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, fc_rate=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = CausalCrossConditionalSelfAttention(embed_dim, block_size, n_head, drop_out_rate)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x):
        """
        x: (B, 51, embed_dim)   51: 1 conditional embed + 50 sequence embed
        """
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))  # Feed forward
        return x


class CrossCondTransBase(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16,  # T2M-GPT default: 51. same to the output motion sequence length
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()
        self.tok_emb = nn.Embedding(num_vq + 2, embed_dim)
        self.cond_emb = nn.Linear(clip_dim, embed_dim)
        self.pos_embedding = nn.Embedding(block_size, embed_dim)  # learnable position embedding, seems not being used by T2M-GPT.
        self.drop = nn.Dropout(drop_out_rate)
        # transformer block
        self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.pos_embed = pos_encoding.PositionEmbedding(block_size, embed_dim, 0.0, False)

        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):

        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, idx, clip_feature):
        '''
        Input:
        idx: (B, 50)
            50: input motion token length.
            value range: 0~513 (512 vq_num (0~511) + 1 end flag (512) + 1 padding flag (513)
        feat_clip_text: (B, 512)
        '''
        if len(idx) == 0:
            ''' It seems only will be used at inference. So its second dim length can be 1 rather than 51'''
            token_embeddings = self.cond_emb(clip_feature).unsqueeze(1)  # (B, 512) => (B, 1, embed_dim)
        else:
            b, t = idx.size()
            assert t <= self.block_size, "Cannot forward, model block size is exhausted."
            # forward the Trans model
            token_embeddings = self.tok_emb(idx)  # (B, 50) => (B, 50, embed_dim)
            # get (B, 51, embed_dim)
            token_embeddings = torch.cat([self.cond_emb(clip_feature).unsqueeze(1), token_embeddings], dim=1)

        x = self.pos_embed(token_embeddings)  # (B, 51, embed_dim); add the position embeddings
        x = self.blocks(x)

        return x


class CrossCondTransHead(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                block_size=16,  # T2M-GPT default: 51. same to the output motion sequence length
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()

        self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_vq + 1, bias=False)
        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        x = self.blocks(x)  # (B, 51, embed_dim)
        x = self.ln_f(x)  # (B, 51, embed_dim)
        logits = self.head(x)  # (B, 51, 513)  513: num_vq + 1
        return logits

    


        


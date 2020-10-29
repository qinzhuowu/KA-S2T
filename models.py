import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import copy
import math
print_dims = False
class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        # S x B x H
        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        max_len = encoder_outputs.size(0)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        # For each position of encoder outputs
        this_batch_size = encoder_outputs.size(1)
        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, 2 * self.hidden_size)
        attn_energies = self.score(torch.tanh(self.attn(energy_in)))  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask, -1e12)
        attn_energies = self.softmax(attn_energies)
        # Normalize energies to weights in range 0 to 1, resize to B x 1 x S
        return attn_energies.unsqueeze(1)


class AttnDecoderRNN(nn.Module):
    def __init__(
            self, hidden_size, embedding_size, input_size, output_size, n_layers=2, dropout=0.5):
        super(AttnDecoderRNN, self).__init__()

        # Keep for reference
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.em_dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.gru = nn.GRU(hidden_size + embedding_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        # Choose attention model
        self.attn = Attn(hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs, seq_mask):
        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seqs)
        embedded = self.em_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.embedding_size)  # S=1 x B x N

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(last_hidden[-1].unsqueeze(0), encoder_outputs, seq_mask)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x S=1 x N

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(torch.cat((embedded, context.transpose(0, 1)), 2), last_hidden)

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        output = self.out(torch.tanh(self.concat(torch.cat((rnn_output.squeeze(0), context.squeeze(1)), 1))))

        # Return final output, hidden state
        return output, hidden


class TreeNode:  # the class save the tree node
    def __init__(self, embedding, left_flag=False):
        self.embedding = embedding
        self.left_flag = left_flag


class Score(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Score, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, num_embeddings, num_mask=None):
        max_len = num_embeddings.size(1)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[1] = max_len
        hidden = hidden.repeat(*repeat_dims)  # B x O x H
        # For each position of encoder outputs
        this_batch_size = num_embeddings.size(0)
        energy_in = torch.cat((hidden, num_embeddings), 2).view(-1, self.input_size + self.hidden_size)
        score = self.score(torch.tanh(self.attn(energy_in)))  # (B x O) x 1
        score = score.squeeze(1)
        score = score.view(this_batch_size, -1)  # B x O
        if num_mask is not None:
            score = score.masked_fill_(num_mask, -1e12)
        return score
class Middle_Score(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Middle_Score, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.attn_left = nn.Linear(hidden_size + input_size, hidden_size)
        self.attn_right = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.score_left=nn.Linear(hidden_size, 1, bias=False)
        self.score_right=nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, num_embeddings, num_mask=None):
        # leaf_input.unsqueeze(1) [B,1,2N]  embedding_weight_ [B, max_num_size+len_generate,N]  max_num_size:N1,N2,N3 size  len_generate: 1,3.14 size
        #num_score = Score(leaf_input.unsqueeze(1), embedding_weight_, mask_nums)
        # max_len num_size+gene_size 
        max_len = num_embeddings.size(1)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[1] = max_len
        hidden = hidden.repeat(*repeat_dims)  # B x num_size+gene_size x 2N
        # For each position of encoder outputs
        this_batch_size = num_embeddings.size(0)
        #B*num_size+gene_size *3N
        energy_in = torch.cat((hidden, num_embeddings), 2).view(-1, self.input_size + self.hidden_size)
        score = self.score(torch.tanh(self.attn(energy_in)))  # (B x O) x 1
        score = score.view(this_batch_size, max_len)  # B x O *1
        if num_mask is not None:
            score = score.masked_fill_(num_mask, -1e12)
        score_left = self.score_left(torch.tanh(self.attn_left(energy_in)))  # (B x O) x 1
        score_left = score_left.view(this_batch_size, max_len)  # B x O
        if num_mask is not None:
            score_left = score_left.masked_fill_(num_mask, -1e12)
        score_right = self.score_right(torch.tanh(self.attn_right(energy_in)))  # (B x O) x 1
        score_right = score_right.view(this_batch_size, max_len)  # B x O
        if num_mask is not None:
            score_right = score_right.masked_fill_(num_mask, -1e12)

        score_middle=torch.cat((score.unsqueeze(2), score_left.unsqueeze(2),score_right.unsqueeze(2)), 2) #B*0*3
        return score_middle


class TreeAttn(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeAttn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        max_len = encoder_outputs.size(0)

        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        this_batch_size = encoder_outputs.size(1)

        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, self.input_size + self.hidden_size)

        score_feature = torch.tanh(self.attn(energy_in))
        attn_energies = self.score(score_feature)  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask, -1e12)
        attn_energies = nn.functional.softmax(attn_energies, dim=1)  # B x S

        return attn_energies.unsqueeze(1)


class EncoderSeq(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size,category_size, n_layers=2, dropout=0.5):
        super(EncoderSeq, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)
        self.category_embedding = nn.Embedding(category_size, hidden_size/2, padding_idx=0)
        #self.gru_pade = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

        self.gat_1 = GATLayer(hidden_size, hidden_size, 0.5, 0.1,concat=False )
        self.gat_dense=nn.Linear(hidden_size*2,hidden_size)


        self.gru_pade = nn.GRU(embedding_size, hidden_size/2, n_layers, dropout=dropout, bidirectional=True)

        self.gat_n = [GraphAttentionLayer(hidden_size/2, hidden_size/2, 0.5, 0.1,concat=False ) for _ in range(8)]
        for i, attention in enumerate(self.gat_n):
            self.add_module('attention_{}'.format(i), attention)
        
        self.pos_embedding = PositionalEncoding(hidden_size/2, 160)
        self.encoder_layers =EncoderLayer(hidden_size, 16, hidden_size, dropout)
        self.encoder_layers2 =EncoderLayer(hidden_size, 4, hidden_size, dropout)

    def forward(self, input_seqs, input_lengths,cate_word_edge,cate_index_input,cate_length,cate_id_match, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        pade_hidden = hidden
        pade_outputs, pade_hidden = self.gru_pade(packed, pade_hidden)
        pade_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs)


        #problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        pade_outputs = pade_outputs[:, :, :self.hidden_size/2] + pade_outputs[:, :, self.hidden_size/2:]  # S x B x H
        #problem_output=F.max_pool1d(pade_outputs.permute(1,2,0), pade_outputs.shape[0]).squeeze(-1) #(batch_size, d_model,S) (batch_size, d_model)

        #encoder_outputs_knowledge=self.gat_1(pade_outputs.transpose(0,1),input_edge_batch)#B*S*H
        #concat_pade_outputs=self.gat_dense(torch.cat((pade_outputs,encoder_outputs_knowledge.transpose(0,1)),2))
        #return concat_pade_outputs, problem_output
        
        ##category_embedded=self.category_embedding(cate_index_input)#B*cate_num*hidden
        #cate_length=[]#B cate_id_match=[]#B*C*[]
        max_cate_len=max(cate_length)
        

        padding_hidden=self.category_embedding(torch.LongTensor([0]).cuda()).squeeze(0)
        max_cate_len = max(cate_length)
        if max_cate_len>0:
            category_embedded_temp=[]
            for idx in range(len(cate_length)):
                idx_cate_len=cate_length[idx]
                if idx_cate_len == 0:
                    category_embedded_temp.append(torch.stack([padding_hidden for cate_idx in range(max_cate_len)],dim=0))#C*H
                else:
                    temp_hidden_category=[]
                    for i in range(idx_cate_len):
                        temp_hidden=[]
                        for j in cate_id_match[idx][i]:
                            temp_hidden.append(pade_outputs[j][idx])#cate*hidden
                        gather_hidden=torch.stack(temp_hidden,dim=0).mean(0)#hidden
                        temp_hidden_category.append(gather_hidden)
                    for i in range(idx_cate_len,max_cate_len):
                        temp_hidden_category.append(padding_hidden)
                    category_embedded_temp.append(torch.stack(temp_hidden_category,0))#C*hidden
            category_embedded = torch.stack(category_embedded_temp,0).detach()#B*S*H

            concat_input_category=torch.cat((pade_outputs.transpose(0,1),category_embedded),1)#B*S+cate_num*hidden
        else:
            concat_input_category=pade_outputs.transpose(0,1)
        
        input_sequence=input_seqs.transpose(0,1)

        encoder_outputs_knowledge = torch.stack([att(concat_input_category,cate_word_edge) for att in self.gat_n], dim=2)#B*S*N*head
        encoder_outputs_knowledge=encoder_outputs_knowledge.mean(2)#B*S*N
        
        #pos_embed= self.pos_embedding(input_lengths,max_cate_len)#B[B,S,H]
        pos_embed= self.pos_embedding(input_lengths,max_cate_len)#B[B,S,H]
        output=torch.cat((encoder_outputs_knowledge,pos_embed),2)#B[B,S,H]
        
        src_mask = (input_sequence != 0).unsqueeze(-2)
        if max_cate_len > 0:
            cat_mask = (cate_index_input != 0).unsqueeze(-2) #B*1*C
            src_cat_mask=torch.cat((src_mask,cat_mask),2)
        else:
            src_cat_mask=src_mask
        output, attention = self.encoder_layers(output, src_cat_mask)#[B,S,H] B*S*S
        
        output=output[:,:input_seqs.size(0),:]
        #problem_output=F.max_pool1d(pade_outputs.permute(1,2,0), pade_outputs.shape[0]).squeeze(-1) #(batch_size, d_model,S) (batch_size, d_model)
        problem_output=F.max_pool1d(output.permute(0,2,1), output.shape[1]).squeeze(-1) #(batch_size, d_model,S) (batch_size, d_model)

        return output.transpose(0,1), problem_output
        '''
        #problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        pade_outputs = pade_outputs[:, :, :self.hidden_size/2] + pade_outputs[:, :, self.hidden_size/2:]  # S x B x H
        #problem_output=F.max_pool1d(pade_outputs.permute(1,2,0), pade_outputs.shape[0]).squeeze(-1) #(batch_size, d_model,S) (batch_size, d_model)

        #encoder_outputs_knowledge=self.gat_1(pade_outputs.transpose(0,1),input_edge_batch)#B*S*H
        #concat_pade_outputs=self.gat_dense(torch.cat((pade_outputs,encoder_outputs_knowledge.transpose(0,1)),2))
        #return concat_pade_outputs, problem_output

        input_sequence=input_seqs.transpose(0,1)

        encoder_outputs_knowledge = torch.stack([att(pade_outputs.transpose(0,1),input_edge_batch) for att in self.gat_n], dim=2)#B*S*N*head
        encoder_outputs_knowledge=encoder_outputs_knowledge.mean(2)#B*S*N
        
        pos_embed= self.pos_embedding(input_lengths)#B[B,S,H]
        output=torch.cat((encoder_outputs_knowledge,pos_embed),2)#B[B,S,H]
        
        src_mask = (input_sequence != 0).unsqueeze(-2)
        output, attention = self.encoder_layers(output, src_mask)#[B,S,H] B*S*S
        
        #problem_output=F.max_pool1d(pade_outputs.permute(1,2,0), pade_outputs.shape[0]).squeeze(-1) #(batch_size, d_model,S) (batch_size, d_model)
        problem_output=F.max_pool1d(output.permute(0,2,1), output.shape[1]).squeeze(-1) #(batch_size, d_model,S) (batch_size, d_model)

        return output.transpose(0,1), problem_output
        '''
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Linear(in_features, out_features, bias=False)
        # self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        # self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        self.a1 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        self.a2 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, graph_input, adj):
        #[B*S*H] [B*S*S]
        h = self.W(graph_input)
        # [batch_size, N, out_features]
        batch_size, N,  _ = h.size()
        middle_result1 = torch.matmul(h, self.a1).expand(-1, -1, N) #B*S*N - B*S*1 - B*S*S B*S*S
        middle_result2 = torch.matmul(h, self.a2).expand(-1, -1, N).transpose(1, 2) ##B*S*S
        e = self.leakyrelu(middle_result1 + middle_result2)
        attention = e.masked_fill(adj == 0, -1e9)
        attention = F.softmax(attention, dim=2)
        #attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, graph_input)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_seq_len):
        """ d_model hidden 512  max_seq_len largedt len
        """
        super(PositionalEncoding, self).__init__()
        # PE matrix
        position_encoding = np.array([
          [pos / pow(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]  for pos in range(max_seq_len)])
        # odd line use sin,even line use cos
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        # first line is all 0 as PAD positional encoding
        # word embedding add UNK as word embedding
        # use PAD to represent PAD position
        position_encoding=torch.FloatTensor(position_encoding)
        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.cat((pad_row, position_encoding),0)
        
        # +1 because adding PAD
        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding,requires_grad=False)
    def forward(self, input_len,category_num):
        """input_len  [BATCH_SIZE]
        """
        max_len = max(input_len)+category_num
        #tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        # pad position add 0  range start by 1 to avoid pad(0)
        input_pos = torch.cuda.LongTensor([list(range(1, len + 1)) + [0] * (max_len - len) for len in input_len])
        return self.position_encoding(input_pos)

def attention(query, key, value, mask=None, dropout=None):
    """
    Compute scaled dot product self attention
        query: (batch_size, h, seq_len, d_k), seq_len can be either src_seq_len or tgt_seq_len
        key: (batch_size, h, seq_len, d_k), seq_len in key, value and mask are the same
        value: (batch_size, h, seq_len, d_k)
        mask: (batch_size, 1, 1, seq_len) or (batch_size, 1, tgt_seq_len, tgt_seq_len) (legacy)
    """
    if print_dims:
        print("{0}: query: type: {1}, shape: {2}".format("attention func", query.type(), query.shape))
        print("{0}: key: type: {1}, shape: {2}".format("attention func", key.type(), key.shape))
        print("{0}: value: type: {1}, shape: {2}".format("attention func", value.type(), value.shape))
        print("{0}: mask: type: {1}, shape: {2}".format("attention func", mask.type(), mask.shape))
    d_k = query.size(-1)

    # scores: (batch_size, h, seq_len, seq_len) for self_attn, (batch_size, h, tgt_seq_len, src_seq_len) for src_attn
    scores = torch.matmul(query, key.transpose(-2, -1)/math.sqrt(d_k)) #B,H,S,S
    # print(query.shape, key.shape, mask.shape, scores.shape)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        #scores = scores.masked_fill(mask, -1e9)
    p_attn = F.softmax(scores, dim=-1)##B,H,S,S
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert model_dim % num_heads == 0
        self.dim_per_head = model_dim//num_heads
        self.h = num_heads
        #self.linears = nn.ModuleList([copy.deepcopy(nn.Linear(model_dim, model_dim)) for i in range(4)])
        self.key_dim=12
        self.value_dim=32
        self.linear_k = nn.Linear(model_dim, self.key_dim * num_heads)
        self.linear_v = nn.Linear(model_dim, self.value_dim * num_heads)
        self.linear_q = nn.Linear(model_dim, self.key_dim * num_heads)

        self.linear_x=nn.Linear(self.value_dim * num_heads,model_dim)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.a = nn.Parameter(torch.Tensor([1]))
        self.b = nn.Parameter(torch.Tensor([1]))
        
    def forward(self, query, key, value, mask=None):
        """
            query: (batch_size, seq_len, d_model), seq_len can be either src_seq_len or tgt_seq_len
            key: (batch_size, seq_len, d_model), seq_len in key, value and mask are the same
            value: (batch_size, seq_len, d_model)
            mask: (batch_size, 1, seq_len) or (batch_size, tgt_seq_len, tgt_seq_len) (legacy)
        """
        if print_dims:
            print("{0}: query: type: {1}, shape: {2}".format(self.__class__.__name__, query.type(), query.shape))
            print("{0}: key: type: {1}, shape: {2}".format(self.__class__.__name__, key.type(), key.shape))
            print("{0}: value: type: {1}, shape: {2}".format(self.__class__.__name__, value.type(), value.shape))
            print("{0}: mask: type: {1}, shape: {2}".format(self.__class__.__name__, mask.type(), mask.shape))
        if mask is not None:
            mask = mask.unsqueeze(1)#B,1,1,S
        nbatches = query.size(0)
        
        # 1) Do all linear projections in batch from d_model to (h, d_k)
        key = self.linear_k(key).view(nbatches, -1, self.h, self.key_dim).transpose(1,2)#B*S*(dim_per_head * num_heads)
        value = self.linear_v(value).view(nbatches, -1, self.h, self.value_dim).transpose(1,2)#B,H,S,dim
        query = self.linear_q(query).view(nbatches, -1, self.h, self.key_dim).transpose(1,2)
        #query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1,2) for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch
        x, p_attn = attention(query, key, value, mask=mask, dropout=self.dropout) # (batch_size, h, seq_len, d_k),#B,H,S,S
        if print_dims:
            print("{0}: x (after attention): type: {1}, shape: {2}".format(self.__class__.__name__, x.type(), x.shape))

        # 3) Concatenate and apply a final linear
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.value_dim)##B,S,H,dim
        x = self.linear_x(x) # (batch_size, seq_len, d_model)
        if print_dims:
            print("{0}: x (after concatenation and linear): type: {1}, shape: {2}".format(self.__class__.__name__, x.type(), x.shape))
        return x,p_attn

class PositionalWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
            x: (batch_size, seq_len, d_model)
        """
        if print_dims:
            print("{0}: x: type: {1}, shape: {2}".format(self.__class__.__name__, x.type(), x.shape))
        return self.w2(self.dropout(F.relu(self.w1(x))))

class LayerNorm(nn.Module):
    "Construct a layernorm module"
    def __init__(self, n_features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(n_features))
        self.b_2 = nn.Parameter(torch.zeros(n_features))
        self.eps = eps
        
    def forward(self, x):
        """
            x: (batch_size, seq_len, d_model)
        """
        if print_dims:
            print("{0}: x: type: {1}, shape: {2}".format(self.__class__.__name__, x.type(), x.shape))
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.a_2 * (x - mean)/(std + self.eps) + self.b_2
class EncoderLayer(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2018, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)
        self.norm = LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, attn_mask=None):
        """norm -> self_attn -> dropout -> add -> 
        norm -> feed_forward -> dropout -> add"""
        # self attention  ##[B,S,H] B*S*S
        norm_inputs=self.norm(inputs)
        context, attention = self.attention(norm_inputs, norm_inputs, norm_inputs, attn_mask)#[B,S,H] B*S*S
        context=self.dropout(context)
        context=inputs+context
        # feed forward network
        output=self.norm(context)
        output = self.feed_forward(output)#[B,S,H] #ff x+output
        output=self.dropout(output)
        output=context+output
        return output, attention


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.out_features=32
        self.fc= nn.Linear(in_features, self.out_features, bias=False)
        self.attn_fc= nn.Linear(2*self.out_features,1, bias=False)
        self.concat = concat
        self.dropout = dropout
        self.alpha = alpha
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, graph_input, adj):
        #[B*S*H] [B*S*S]
        h=self.fc(graph_input)
        # [batch_size, N, out_features]
        batch_size, N, out_dim = h.size()#B,N,H
        a_input=torch.cat([h.unsqueeze(2).repeat(1,1,N,1),h.unsqueeze(1).repeat(1,N,1,1)],dim=3)#B,N,N,2*dim
        attention=self.leakyrelu(self.attn_fc(a_input).squeeze(3))#B,S,S
        attention = attention.masked_fill(adj == 0, -1e9)
        attention = F.softmax(attention, dim=2)#B,S,S
        #attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, graph_input)##B,S,S*#B,S,H  #B,S,H  not use h,use original_input
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class Prediction(nn.Module):
    # a seq2tree decoder with Problem aware dynamic encoding

    def __init__(self, hidden_size, op_nums, input_size, dropout=0.5):
        super(Prediction, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.op_nums = op_nums

        # Define layers
        self.dropout = nn.Dropout(dropout)

        self.embedding_weight = nn.Parameter(torch.randn(1, input_size, hidden_size))

        # for Computational symbols and Generated numbers
        self.concat_l = nn.Linear(hidden_size, hidden_size)
        self.concat_r = nn.Linear(hidden_size * 2, hidden_size)
        self.concat_lg = nn.Linear(hidden_size, hidden_size)
        self.concat_rg = nn.Linear(hidden_size * 2, hidden_size)

        self.ops = nn.Linear(hidden_size * 2, op_nums)

        self.attn = TreeAttn(hidden_size, hidden_size)
        self.score = Score(hidden_size * 2, hidden_size)
        self.concat_encoder_outputs = nn.Linear(hidden_size * 2, hidden_size)

        self.middle_score=Middle_Score(hidden_size * 2, hidden_size)
        self.middle_ops = nn.Linear(hidden_size * 2, op_nums)
        self.middle_ops_left = nn.Linear(hidden_size * 2, op_nums)
        self.middle_ops_right = nn.Linear(hidden_size * 2, op_nums)

    def forward(self, node_stacks, left_childs, encoder_outputs, num_pades, padding_hidden, seq_mask, mask_nums):
        current_embeddings = []

        for st in node_stacks:
            if len(st) == 0:
                current_embeddings.append(padding_hidden)
            else:
                current_node = st[-1]
                current_embeddings.append(current_node.embedding)

        current_node_temp = []
        for l, c in zip(left_childs, current_embeddings):
            if l is None:
                c = self.dropout(c)
                g = torch.tanh(self.concat_l(c))
                t = torch.sigmoid(self.concat_lg(c))
                current_node_temp.append(g * t)
            else:
                ld = self.dropout(l)
                c = self.dropout(c)
                g = torch.tanh(self.concat_r(torch.cat((ld, c), 1)))
                t = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))
                current_node_temp.append(g * t)

        current_node = torch.stack(current_node_temp)

        current_embeddings = self.dropout(current_node)

        current_attn = self.attn(current_embeddings.transpose(0, 1), encoder_outputs, seq_mask)
        current_context = current_attn.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N
        
        #encoder_outputs_knowledge=input_edge_batch.bmm(encoder_outputs.transpose(0, 1)) # B x S*S  B x S x H B x S x H
        #concat_encoder_outputs=torch.cat((encoder_outputs, encoder_outputs_knowledge.transpose(0,1)), dim=2)
        #current_attn = self.attn(current_embeddings.transpose(0, 1), concat_encoder_outputs, seq_mask) # B x S
        #current_context = current_attn.bmm(concat_encoder_outputs.transpose(0, 1))  #B x S S*B*N  B x 1 x N
        #current_context=self.concat_encoder_outputs(current_context)
        # the information to get the current quantity
        batch_size = current_embeddings.size(0)
        # predict the output (this node corresponding to output(number or operator)) with PADE

        repeat_dims = [1] * self.embedding_weight.dim()
        repeat_dims[0] = batch_size
        embedding_weight = self.embedding_weight.repeat(*repeat_dims)  # B x input_size x N
        embedding_weight = torch.cat((embedding_weight, num_pades), dim=1)  # B x O x N

        #embedding_weight=embedding_weight*num_score_constraints.unsqueeze(2)

        leaf_input = torch.cat((current_node, current_context), 2)
        leaf_input = leaf_input.squeeze(1)
        leaf_input = self.dropout(leaf_input)

        # p_leaf = nn.functional.softmax(self.is_leaf(leaf_input), 1)
        # max pooling the embedding_weight
        embedding_weight_ = self.dropout(embedding_weight)
        num_score = self.score(leaf_input.unsqueeze(1), embedding_weight_, mask_nums)#gen_size+num_size

        # num_score = nn.functional.softmax(num_score, 1)

        op = self.ops(leaf_input)

        num_middle_score=self.middle_score(leaf_input.unsqueeze(1), embedding_weight_, mask_nums)
        
        op_middle=self.middle_ops(leaf_input).unsqueeze(2)
        op_middle_left=self.middle_ops_left(leaf_input).unsqueeze(2)
        op_middle_right=self.middle_ops_right(leaf_input).unsqueeze(2)

        op_middle=torch.cat((op_middle,op_middle_left, op_middle_right), dim=2)#B*op_nums*3

        # return p_leaf, num_score, op, current_embeddings, current_attn

        return num_score, op, current_node, current_context, embedding_weight,num_middle_score,op_middle


class GenerateNode(nn.Module):
    def __init__(self, hidden_size, op_nums, output_vocab_len,embedding_size, dropout=0.5):
        super(GenerateNode, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(op_nums, hidden_size)
        self.em_dropout = nn.Dropout(dropout)
        self.generate_l = nn.Linear(hidden_size * 2 + hidden_size+hidden_size*3, hidden_size)
        self.generate_r = nn.Linear(hidden_size * 2 + hidden_size+hidden_size*3, hidden_size)
        self.generate_lg = nn.Linear(hidden_size * 2 + hidden_size+hidden_size*3, hidden_size)
        self.generate_rg = nn.Linear(hidden_size * 2 + hidden_size+hidden_size*3, hidden_size)

        #self.generate_l = nn.Linear(hidden_size * 2 + hidden_size, hidden_size)
        #self.generate_r = nn.Linear(hidden_size * 2 + hidden_size, hidden_size)

        self.embeddings_middle=nn.Embedding(output_vocab_len,hidden_size)
        self.embeddings_middle_left=nn.Embedding(output_vocab_len,hidden_size)
        self.embeddings_middle_right=nn.Embedding(output_vocab_len,hidden_size)

    def forward(self, node_embedding, node_label, current_context,outputs_middle_predict):
        outputs_middle_self,outputs_middle_left,outputs_middle_right=outputs_middle_predict.split(1,2)
        middle_self_label=self.em_dropout(self.embeddings_middle(torch.argmax(outputs_middle_self.squeeze(2),dim=1)))
        middle_left_label=self.em_dropout(self.embeddings_middle_left(torch.argmax(outputs_middle_self.squeeze(2),dim=1)))
        middle_right_label=self.em_dropout(self.embeddings_middle_left(torch.argmax(outputs_middle_self.squeeze(2),dim=1)))
        

        node_label_ = self.embeddings(node_label)
        node_label = self.em_dropout(node_label_)
        node_embedding = node_embedding.squeeze(1)
        current_context = current_context.squeeze(1)
        node_embedding = self.em_dropout(node_embedding)
        current_context = self.em_dropout(current_context)

        #l_child = torch.tanh(self.generate_l(torch.cat((node_embedding, current_context, node_label), 1)))
        #r_child = torch.tanh(self.generate_r(torch.cat((node_embedding, current_context, node_label), 1)))
        l_child = torch.tanh(self.generate_l(torch.cat((node_embedding, current_context, node_label,middle_self_label,middle_left_label,middle_right_label), 1)))
        #l_child_g = torch.sigmoid(self.generate_lg(torch.cat((node_embedding, current_context, node_label,middle_self_label,middle_left_label,middle_right_label), 1)))
        r_child = torch.tanh(self.generate_r(torch.cat((node_embedding, current_context, node_label,middle_self_label,middle_left_label,middle_right_label), 1)))
        #r_child_g = torch.sigmoid(self.generate_rg(torch.cat((node_embedding, current_context, node_label,middle_self_label,middle_left_label,middle_right_label), 1)))
        #l_child = l_child * l_child_g
        #r_child = r_child * r_child_g
        return l_child, r_child, node_label_
class Merge(nn.Module):
    def __init__(self, hidden_size, embedding_size, dropout=0.5):
        super(Merge, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.em_dropout = nn.Dropout(dropout)
        self.gcn = nn.Linear(hidden_size, hidden_size)
        self.gcn1 = nn.Linear(hidden_size, hidden_size)

    def forward(self, tree_embed_mat, A_matrix):
        ##t*1*H,t*t
        A_matrix=A_matrix+torch.eye(A_matrix.size(0)).cuda()
        d=A_matrix.sum(1)
        D=torch.diag(torch.pow(d,-1))
        A=D.mm(A_matrix)
        tree_embed_mat=self.em_dropout(tree_embed_mat.squeeze(0))#1*t*H

        new_tree_embed_mat=nn.functional.relu(self.gcn(A.mm(tree_embed_mat)))
        new_tree_embed_mat=nn.functional.relu(self.gcn1(A.mm(new_tree_embed_mat)))
        return new_tree_embed_mat#t*H
'''
class Merge(nn.Module):
    def __init__(self, hidden_size, embedding_size, dropout=0.5):
        super(Merge, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.em_dropout = nn.Dropout(dropout)
        self.merge = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.merge_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, sub_tree_1, sub_tree_2):
        sub_tree_1 = self.em_dropout(sub_tree_1)
        sub_tree_2 = self.em_dropout(sub_tree_2)
        node_embedding = self.em_dropout(node_embedding)

        sub_tree = torch.tanh(self.merge(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree_g = torch.sigmoid(self.merge_g(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree = sub_tree * sub_tree_g
        return sub_tree
'''
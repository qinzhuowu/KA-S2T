#coding:utf-8
from masked_cross_entropy import *
from pre_data import *
from expressions_transfer import *
from models import *
import math
import torch
import torch.optim
import torch.nn.functional as f
import time

from torch.autograd import Variable
MAX_OUTPUT_LENGTH = 45
MAX_INPUT_LENGTH = 120
USE_CUDA = torch.cuda.is_available()


class Beam:  # the class save the beam node
    def __init__(self, score, input_var, hidden, all_output):
        self.score = score
        self.input_var = input_var
        self.hidden = hidden
        self.all_output = all_output


def time_since(s):  # compute time
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)


def generate_rule_mask(decoder_input, nums_batch, word2index, batch_size, nums_start, copy_nums, generate_nums,
                       english):
    rule_mask = torch.FloatTensor(batch_size, nums_start + copy_nums).fill_(-float("1e12"))
    if english:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + \
                      [word2index["("]] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start:
                res += [word2index[")"], word2index["+"], word2index["-"],
                        word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] in generate_nums:
                res += [word2index[")"], word2index["+"], word2index["-"],
                        word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] == word2index["("]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] +\
                  [word2index["("]] + generate_nums
            elif decoder_input[i] == word2index[")"]:
                res += [word2index[")"], word2index["+"], word2index["-"],
                        word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + [word2index["("]] + generate_nums
            for j in res:
                rule_mask[i, j] = 0
    else:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + \
                      [word2index["["], word2index["("]] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [word2index["]"], word2index[")"], word2index["+"],
                        word2index["-"], word2index["/"], word2index["^"],
                        word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] == word2index["["] or decoder_input[i] == word2index["("]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] +\
                  [word2index["("]] + generate_nums
            elif decoder_input[i] == word2index[")"]:
                res += [word2index["]"], word2index[")"], word2index["+"],
                        word2index["-"], word2index["/"], word2index["^"],
                        word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["]"]:
                res += [word2index["+"], word2index["*"], word2index["-"], word2index["/"], word2index["EOS"]]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"],
                                      word2index["*"], word2index["^"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] +\
                  [word2index["["], word2index["("]] + generate_nums
            for j in res:
                rule_mask[i, j] = 0
    return rule_mask


def generate_pre_tree_seq_rule_mask(decoder_input, nums_batch, word2index, batch_size, nums_start, copy_nums,
                                    generate_nums, english):
    rule_mask = torch.FloatTensor(batch_size, nums_start + copy_nums).fill_(-float("1e12"))
    if english:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                      [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]
            for j in res:
                rule_mask[i, j] = 0
    else:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                      [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"]]
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["EOS"],
                        word2index["^"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"],
                                      word2index["^"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"]]
            for j in res:
                rule_mask[i, j] = 0
    return rule_mask


def generate_post_tree_seq_rule_mask(decoder_input, nums_batch, word2index, batch_size, nums_start, copy_nums,
                                     generate_nums, english):
    rule_mask = torch.FloatTensor(batch_size, nums_start + copy_nums).fill_(-float("1e12"))
    if english:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums +\
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            for j in res:
                rule_mask[i, j] = 0
    else:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"],
                                      word2index["^"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"],
                        word2index["EOS"]
                        ]
            for j in res:
                rule_mask[i, j] = 0
    return rule_mask


def generate_tree_input(target, decoder_output, nums_stack_batch, num_start, unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    target_input = copy.deepcopy(target)
    for i in range(len(target)):
        if target[i] == unk:
            num_stack = nums_stack_batch[i].pop()
            max_score = -float("1e12")
            for num in num_stack:
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]
        if target_input[i] >= num_start:
            target_input[i] = 0
    return torch.LongTensor(target), torch.LongTensor(target_input)


def generate_decoder_input(target, decoder_output, nums_stack_batch, num_start, unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    if USE_CUDA:
        decoder_output = decoder_output.cpu()
    for i in range(target.size(0)):
        if target[i] == unk:
            num_stack = nums_stack_batch[i].pop()
            max_score = -float("1e12")
            for num in num_stack:
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]
    return target


def mask_num(encoder_outputs, decoder_input, embedding_size, nums_start, copy_nums, num_pos):
    # mask the decoder input number and return the mask tensor and the encoder position Hidden vector
    up_num_start = decoder_input >= nums_start
    down_num_end = decoder_input < (nums_start + copy_nums)
    num_mask = up_num_start == down_num_end
    num_mask_encoder = num_mask < 1
    num_mask_encoder = num_mask_encoder.unsqueeze(1)  # ByteTensor size: B x 1
    repeat_dims = [1] * num_mask_encoder.dim()
    repeat_dims[1] = embedding_size
    num_mask_encoder = num_mask_encoder.repeat(*repeat_dims)  # B x 1 -> B x Decoder_embedding_size

    all_embedding = encoder_outputs.transpose(0, 1).contiguous()
    all_embedding = all_embedding.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
    indices = decoder_input - nums_start
    indices = indices * num_mask.long()  # 0 or the num pos in sentence
    indices = indices.tolist()
    for k in range(len(indices)):
        indices[k] = num_pos[k][indices[k]]
    indices = torch.LongTensor(indices)
    if USE_CUDA:
        indices = indices.cuda()
    batch_size = decoder_input.size(0)
    sen_len = encoder_outputs.size(0)
    batch_num = torch.LongTensor(range(batch_size))
    batch_num = batch_num * sen_len
    if USE_CUDA:
        batch_num = batch_num.cuda()
    indices = batch_num + indices
    num_encoder = all_embedding.index_select(0, indices)
    return num_mask, num_encoder, num_mask_encoder


def out_equation(test, output_lang, num_list, num_stack=None):
    test = test[:-1]
    max_index = len(output_lang.index2word) - 1
    test_str = ""
    for i in test:
        if i < max_index:
            c = output_lang.index2word[i]
            if c == "^":
                test_str += "**"
            elif c == "[":
                test_str += "("
            elif c == "]":
                test_str += ")"
            elif c[0] == "N":
                if int(c[1:]) >= len(num_list):
                    return None
                x = num_list[int(c[1:])]
                if x[-1] == "%":
                    test_str += "(" + x[:-1] + "/100" + ")"
                else:
                    test_str += x
            else:
                test_str += c
        else:
            if len(num_stack) == 0:
                print(test_str, num_list)
                return ""
            n_pos = num_stack.pop()
            test_str += num_list[n_pos[0]]
    return test_str


def compute_prefix_tree_result(test_res, test_tar, output_lang, num_list, num_stack):
    # print(test_res, test_tar)

    if len(num_stack) == 0 and test_res == test_tar:
        return True, True, out_expression_list(test_res, output_lang, num_list), out_expression_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    test = out_expression_list(test_res, output_lang, num_list)
    tar = out_expression_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    # print(test, tar)
    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(compute_prefix_expression(test) - compute_prefix_expression(tar)) < 1e-4:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar


def compute_postfix_tree_result(test_res, test_tar, output_lang, num_list, num_stack):
    # print(test_res, test_tar)

    if len(num_stack) == 0 and test_res == test_tar:
        return True, True, test_res, test_tar
    test = out_expression_list(test_res, output_lang, num_list)
    tar = out_expression_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    # print(test, tar)
    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(compute_postfix_expression(test) - compute_postfix_expression(tar)) < 1e-4:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar


def compute_result(test_res, test_tar, output_lang, num_list, num_stack):
    if len(num_stack) == 0 and test_res == test_tar:
        return True, True
    test = out_equation(test_res, output_lang, num_list)
    tar = out_equation(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    if test is None:
        return False, False
    if test == tar:
        return True, True
    try:
        if abs(eval(test) - eval(tar)) < 1e-4:
            return True, False
        else:
            return False, False
    except:
        return False, False


def get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size, hidden_size):
    indices = list()
    sen_len = encoder_outputs.size(0)
    masked_index = []
    temp_1 = [1 for _ in range(hidden_size)]
    temp_0 = [0 for _ in range(hidden_size)]
    for b in range(batch_size):
        for i in num_pos[b]:
            indices.append(i + b * sen_len)
            masked_index.append(temp_0)
        indices += [0 for _ in range(len(num_pos[b]), num_size)]
        masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
    indices = torch.LongTensor(indices)
    masked_index = torch.ByteTensor(masked_index)
    masked_index = masked_index.view(batch_size, num_size, hidden_size)
    if USE_CUDA:
        indices = indices.cuda()
        masked_index = masked_index.cuda()
    all_outputs = encoder_outputs.transpose(0, 1).contiguous() #S*B*H,B*S*H
    all_embedding = all_outputs.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
    all_num = all_embedding.index_select(0, indices)
    all_num = all_num.view(batch_size, num_size, hidden_size)
    return all_num.masked_fill_(masked_index, 0.0),indices,masked_index#(B*num*H)


def copy_list(l):
    r = []
    if len(l) == 0:
        return r
    for i in l:
        if type(i) is list:
            r.append(copy_list(i))
        else:
            r.append(i)
    return r


class TreeBeam:  # the class save the beam node
    def __init__(self, score, node_stack, embedding_stack, left_childs, out):
        self.score = score
        self.embedding_stack = copy_list(embedding_stack)
        self.node_stack = copy_list(node_stack)
        self.left_childs = copy_list(left_childs)
        self.out = copy.deepcopy(out)


class TreeEmbedding:  # the class save the tree
    def __init__(self, embedding, terminal=False):
        self.embedding = embedding
        self.terminal = terminal

#def evaluate_tree(input_batch, input_length, generate_nums, encoder, predict, generate, merge, output_lang, num_pos,
#                  beam_size=5, english=False, max_length=MAX_OUTPUT_LENGTH):
# train_tree有 traget信息，stack信息 optimizer信息
#evaluate有 max_length=MAX_OUTPUT_LENGTH  45
def train_tree(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
               encoder, predict, generate, merge, encoder_optimizer, predict_optimizer, generate_optimizer,
               merge_optimizer, input_lang,output_lang, num_pos,  category_index_batch,category_match_batch,
               output_middle_batch,input_edge_batch,hownet_dict_vocab,english=False):
    # sequence mask for attention
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)

    num_mask = []
    max_num_size = max(num_size_batch) + len(generate_nums)
    for i in num_size_batch:
        d = i + len(generate_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.ByteTensor(num_mask)

    unk = output_lang.word2index["UNK"]
    add=output_lang.word2index["+"]
    sub=output_lang.word2index["-"]
    mul=output_lang.word2index["*"]
    div=output_lang.word2index["/"]
    exp=output_lang.word2index["^"]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)

    target = torch.LongTensor(target_batch).transpose(0, 1)

    output_middle=torch.LongTensor(output_middle_batch).transpose(0, 1)#B*out_len*3
    #[[u'/', u'*', 'N2'], [u'*', 'N0', 'N1'], ['N0', 'N0', 'N0'], ['N1', 'N1', 'N1'], ['N2', 'N2', 'N2']]
    #[[2, 0, 9], [0, 7, 8], [7, 7, 7], [8, 8, 8], [9, 9, 9]]

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)

    copy_num_len = [len(_) for _ in num_pos]
    num_size = max(copy_num_len)
    '''
    constraint_matrix=[]#5*ge+num*ge+num
    for i in range(5):
        num_size_matrix=[]
        for j in range(num_size+len(generate_nums)):
            num_size_matrix.append([1]*(num_size+len(generate_nums)) )
        constraint_matrix.append(num_size_matrix)
    '''
    '''
    for idx1 in range(len(unit_list_batch)):
        for idx2 in range(len(unit_list_batch)):
            if idx1!=idx2 and unit_list_batch[idx1]!="" and unit_list_batch[idx2]!="" and unit_list_batch[idx1]==unit_list_batch[idx2]:
                constraint_matrix[mul][idx1+len(generate_nums)][idx2+ len(generate_nums)]=0
                constraint_matrix[exp][idx1+len(generate_nums)][idx2+ len(generate_nums)]=0
            elif idx1!=idx2 and unit_list_batch[idx1]!="" and unit_list_batch[idx2]!="" and unit_list_batch[idx1]!=unit_list_batch[idx2]:
                constraint_matrix[add][idx1+len(generate_nums)][idx2+ len(generate_nums)]=0
                constraint_matrix[sub][idx1+len(generate_nums)][idx2+ len(generate_nums)]=0
    '''
    #unit_list_batch= torch.FloatTensor(unit_list_batch)
    
    max_category_num=0
    for category_index_list in category_index_batch:
        if len(category_index_list) > max_category_num:
            max_category_num=len(category_index_list)
    
    cate_index_input=[]#B*cate_num
    for category_index_list in category_index_batch:
        cate_index_input.append(category_index_list+[0 for _ in range(max_category_num-len(category_index_list))])

    cate_word_edge=[]#B*cate+seq*cate+seq
    for i in input_length:
        temp_edge_matrix=[]
        for j in range(max_len+max_category_num):
            temp_edge_matrix.append([0 for _ in range(max_len+max_category_num)])
        cate_word_edge.append(temp_edge_matrix)

    for i in range(len(input_length)):
        category_match_list=category_match_batch[i]
        for j in range(input_length[i]):
            cate_word_edge[i][j][j]=1
        for j in range(len(category_match_list)):
            category_match_word=category_match_list[j]#[0, 3, 7, 9, 13]
            cate_id=max_len+j
            cate_word_edge[i][cate_id][cate_id]=1
            for word_id in category_match_word:
                cate_word_edge[i][word_id][cate_id]=1
                cate_word_edge[i][cate_id][word_id]=1
        
    for i in range(len(input_length)):
        for j1 in range(input_length[i]):
            for j2 in range(input_length[i]):
                word1= input_lang.index2word[input_batch[i][j1]]
                word2= input_lang.index2word[input_batch[i][j2]]
                if word1 in hownet_dict_vocab:
                    cate1 = hownet_dict_vocab[word1]
                    if len(cate1) >0 and word2==word1 and len(word1)>3 and word1!="NUM":
                        cate_word_edge[i][j1][j2]=1
                        cate_word_edge[i][j2][j1]=1
    cate_id_match=[]#B*C*[]
    cate_length=[]#B
    
    for i in range(len(input_length)):
        
        category_match_list=category_match_batch[i]
        cate_length.append(len(category_match_list))
        cate_id_list=[]
        for j in range(len(category_match_list)):
            cate_id_list.append(category_match_list[j])
        for j in range(len(category_match_list),max_category_num):
            cate_id_list.append([max_len-1])
        cate_id_match.append(cate_id_list)
        

    '''
    if cate_word_edge!=input_edge_batch:
        for i in range(len(input_length)):
            if cate_word_edge[i]!=input_edge_batch[i]:
                print("################################")
                print(" ".join(indexes_to_sentence(input_lang,input_batch[i])))
                print(" ".join(indexes_to_sentence(output_lang,target_batch[0])))
                print(len(cate_word_edge[i]))
                print(len(input_edge_batch[i]))
                cate_word_edge_list=[]
                for j1 in range(len(cate_word_edge[i])):
                    for j2 in range(len(cate_word_edge[i])):
                        if cate_word_edge[i][j1][j2]==1:
                            temp_list=[j1,j2]
                            cate_word_edge_list.append(temp_list)
                input_edge_batch_list=[]
                for j1 in range(len(input_edge_batch[i])):
                    for j2 in range(len(input_edge_batch[i])):
                        if input_edge_batch[i][j1][j2]==1:
                            temp_list=[j1,j2]
                            input_edge_batch_list.append(temp_list)
                print(cate_word_edge_list)
                print(input_edge_batch_list)
                print(category_index_batch[i])
                print(category_match_batch[i])
    '''

    cate_index_input=torch.LongTensor(cate_index_input)##B*cate_num
    cate_word_edge=torch.FloatTensor(cate_word_edge)##B*cate+seq*cate+seq

    input_edge_batch= torch.FloatTensor(input_edge_batch)
    encoder.train()
    predict.train()
    generate.train()
    merge.train()

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
        input_edge_batch=input_edge_batch.cuda()
        #input_edge_batch=input_edge_batch.cuda()
        cate_index_input=cate_index_input.cuda()
        cate_word_edge=cate_word_edge.cuda()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    merge_optimizer.zero_grad()
    # Run words through encoder

    encoder_outputs, problem_output = encoder(input_var, input_length,cate_word_edge,cate_index_input,cate_length,cate_id_match)#B*cate+seq*cate+seq
    
    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    max_target_length = max(target_length)

    all_node_outputs = []
    all_middle_outputs=[]
    # all_leafs = []

    all_nums_encoder_outputs,indices,masked_index = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
                                                              encoder.hidden_size)

    num_start = output_lang.num_start
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]

    num_score_constraints=[[]]
    #print("****************")
    #print(num_size)
    #print(len(generate_nums))
    for t in range(max_target_length):
        num_score_constraints=[]
        '''
        for idx in range(batch_size):
            num_score_constraints.append([1]*(num_size+len(generate_nums)) )#B*(num+gen)
        if t>0:
            for idx,i_left in zip(range(batch_size),target[t-1].tolist()):
                if 3 in cons_mode:
                    if i_left == add or i_left==sub:
                        for rule3_idx in rule3_list_batch[idx]:
                            num_score_constraints[idx][rule3_idx+len(generate_nums)]=0
                if t>1:
                    i_op=target[t-2].tolist()[idx]
                    if i_op<num_start and i_left>=num_start+len(generate_nums):
                        idx_left=i_left-num_start-len(generate_nums)
                        if unit_list_batch[idx][idx_left]!="":
                            if 2 in cons_mode:
                                if i_op == add or i_op==sub:
                                    for idx_right in range(len(unit_list_batch[idx])):
                                        if idx_right!=idx_left and unit_list_batch[idx][idx_right]!="" and unit_list_batch[idx][idx_left]!=unit_list_batch[idx][idx_right]:
                                            #print(idx_right)
                                            num_score_constraints[idx][idx_right+len(generate_nums)]=0
                            if 1 in cons_mode:
                                if i_op==mul or i_op==exp:
                                    for idx_right in range(len(unit_list_batch[idx])):
                                        if idx_right!=idx_left and unit_list_batch[idx][idx_right]!="" and unit_list_batch[idx][idx_left]==unit_list_batch[idx][idx_right]:
                                            num_score_constraints[idx][idx_right+len(generate_nums)]=0
                            if 3 in cons_mode:
                                if i_op == add or i_op==sub:          
                                    for rule3_idx in rule3_list_batch[idx]:
                                        num_score_constraints[idx][rule3_idx+len(generate_nums)]=0
        num_score_constraints=torch.FloatTensor(num_score_constraints).cuda()
        '''
        num_score, op, current_embeddings, current_context, current_nums_embeddings,num_middle_score,op_middle = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

        # all_leafs.append(p_leaf)

        outputs = torch.cat((op, num_score), 1)
        all_node_outputs.append(outputs)
        # B*op_mum+gene_num+num_num*3
        outputs_middle_predict=torch.cat((op_middle,num_middle_score), 1)
        # max_traget_length, b*N*3
        all_middle_outputs.append(outputs_middle_predict)
        
        target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, nums_stack_batch, num_start, unk)
        target[t] = target_t
        if USE_CUDA:
            generate_input = generate_input.cuda()
        left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context,outputs_middle_predict)

        left_childs = []
        for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                               node_stacks, target[t].tolist(), embeddings_stacks):
            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                left_childs.append(None)
                continue

            temp_node_stack=[]
            A_matrix=[[0 for _ in range(t+1)] for _ in range(t+1)]
            for t_idx in range(t+1):
                token_idx=target_batch[idx][t_idx]
                if len(temp_node_stack)!=0:
                    parent_idx=temp_node_stack.pop()
                    A_matrix[parent_idx][t_idx]=1
                    A_matrix[t_idx][parent_idx]=1
                if token_idx<num_start:
                    temp_node_stack.append(t_idx)
                    temp_node_stack.append(t_idx)


            if len(temp_node_stack)!=0:
                parent_idx=temp_node_stack.pop()
            else:
                parent_idx=0


            if i < num_start:
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))
                o.append(node_label[idx].unsqueeze(0))
                #o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
            else:
                current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
                o.append(current_num)
                #while len(o) > 0 and o[-1].terminal:
                #    sub_stree = o.pop()
                #    op = o.pop()
                #    current_num = merge(op.embedding, sub_stree.embedding, current_num)
                #o.append(TreeEmbedding(current_num, True))
            #if len(o) > 0 and o[-1].terminal:
            #    left_childs.append(o[-1].embedding)
            #else:
            #left_childs.append(None)
            tree_embed_mat=copy_list(o)#S*H
            new_tree_embed_mat=merge(torch.stack(tree_embed_mat, dim=1),torch.FloatTensor(A_matrix).cuda())#1*t*H,t*t
            tree_embed_list=new_tree_embed_mat.split(1)
            left_childs.append(tree_embed_list[parent_idx])
            for t_idx in range(t+1):
                o[t_idx]=tree_embed_list[t_idx]
    # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
    all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

    target = target.transpose(0, 1).contiguous()

    #B*S*N*3 3*B*S*N
    all_middle_outputs=torch.stack(all_middle_outputs,dim=1).permute(3,0,1,2)
    all_middle_outputs=all_middle_outputs.contiguous().view(3*batch_size,max_target_length,-1)
    #all_middle,all_middle_left,all_middle_right=all_middle_outputs.split(1,dim=3)
    #all_middle=all_middle.squeeze(3)
    #all_middle_left=all_middle_left.squeeze(3)
    #all_middle_right=all_middle_right.squeeze(3)
    #S*B*3 3*B*S
    output_middle=output_middle.permute(2,1,0).contiguous().view(3*batch_size,max_target_length)
    #output_middle_curr,output_middle_left,output_middle_right=output_middle.split(1,dim=2)
    #output_middle_curr=output_middle_curr.squeeze(2)
    #output_middle_left=output_middle_left.squeeze(2)
    #output_middle_right=output_middle_right.squeeze(2)
    middle_target_length=[]
    middle_target_length=target_length+target_length+target_length


    if USE_CUDA:
        # all_leafs = all_leafs.cuda()
        all_node_outputs = all_node_outputs.cuda()
        target = target.cuda()
        #B*S*N*3
        all_middle_outputs=all_middle_outputs.cuda()
        #all_middle=all_middle.cuda()
        #all_middle_left=all_middle_left.cuda()
        #all_middle_right=all_middle_right.cuda()
        #B*S*3
        output_middle=output_middle.cuda()
        #output_middle_curr=output_middle_curr.cuda()
        #output_middle_left=output_middle_left.cuda()
        #output_middle_right=output_middle_right.cuda()

    # op_target = target < num_start
    # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
    loss1 = masked_cross_entropy(all_node_outputs, target, target_length)
    loss2=masked_cross_entropy(all_middle_outputs, output_middle, middle_target_length)
    # loss = loss_0 + loss_1
    loss=loss1+loss2
    loss.backward()
    # clip the grad
    # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(predict.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(generate.parameters(), 5)

    # Update parameters with optimizers
    encoder_optimizer.step()
    predict_optimizer.step()
    generate_optimizer.step()
    merge_optimizer.step()
    return loss.item()  # , loss_0.item(), loss_1.item()


def evaluate_tree(input_batch, input_length, generate_nums, encoder, predict, generate, merge, input_lang,output_lang, 
    num_pos,category_index_list,category_match_list,output_middle_batch,input_edge_batch,hownet_dict_vocab,beam_size=5, english=False, max_length=MAX_OUTPUT_LENGTH):

    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).unsqueeze(1)

    num_mask = torch.ByteTensor(1, len(num_pos) + len(generate_nums)).fill_(0)

    max_category_num=len(category_index_list)


    cate_index_input=[]#B*cate_num    
    if len(category_index_list)==0:
        category_index_list=[0]
    cate_index_input.append(category_index_list)

    cate_word_edge=[]#B*cate+seq*cate+seq
    temp_edge_matrix=[]
    for j in range(input_length+max_category_num):
        temp_edge_matrix.append([0 for _ in range(input_length+max_category_num)])
    cate_word_edge.append(temp_edge_matrix)

    for j in range(input_length):
        cate_word_edge[0][j][j]=1

    
    for j in range(len(category_match_list)):
        category_match_word=category_match_list[j]#[0, 3, 7, 9, 13]
        cate_id=input_length+j
        cate_word_edge[0][cate_id][cate_id]=1
        for word_id in category_match_word:
            cate_word_edge[0][word_id][cate_id]=1
            cate_word_edge[0][cate_id][word_id]=1
    
    for j1 in range(input_length):
        for j2 in range(input_length):
            word1= input_lang.index2word[input_batch[j1]]
            word2= input_lang.index2word[input_batch[j2]]
            if word1 in hownet_dict_vocab:
                cate1 = hownet_dict_vocab[word1]
                if len(cate1) >0 and word2==word1 and len(word1)>3 and word1!="NUM":
                    cate_word_edge[0][j1][j2]=1
                    cate_word_edge[0][j2][j1]=1

    input_edge_batch= torch.FloatTensor(input_edge_batch).unsqueeze(0)
    cate_index_input=torch.LongTensor(cate_index_input)##B*cate_num
    cate_word_edge=torch.FloatTensor(cate_word_edge)##B*cate+seq*cate+seq
    
    cate_id_match=[]#B*C*[]
    cate_length=[]#B

    cate_length.append(len(category_match_list))
    cate_id_list=[]
    for j in range(len(category_match_list)):
        cate_id_list.append(category_match_list[j])
    cate_id_match.append(cate_id_list)
    # Set to not-training mode to disable dropout

    # Set to not-training mode to disable dropout
    encoder.eval()
    predict.eval()
    generate.eval()
    merge.eval()

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    add=output_lang.word2index["+"]
    sub=output_lang.word2index["-"]
    mul=output_lang.word2index["*"]
    div=output_lang.word2index["/"]
    exp=output_lang.word2index["^"]
    batch_size = 1

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
        input_edge_batch=input_edge_batch.cuda()
        cate_index_input=cate_index_input.cuda()
        cate_word_edge=cate_word_edge.cuda()
    # Run words through encoder
    # 
    encoder_outputs, problem_output = encoder(input_var, [input_length],cate_word_edge,cate_index_input,cate_length,cate_id_match)

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    num_size = len(num_pos)
    all_nums_encoder_outputs,indices,masked_index = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
                                                              encoder.hidden_size)
    num_start = output_lang.num_start
    # B x P x N
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]

    beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]
    #print("********************")
    #print(unit_list_batch)
    #print(num_size)
    for t in range(max_length):
        current_beams = []
        while len(beams) > 0:
            b = beams.pop()
            if len(b.node_stack[0]) == 0:
                current_beams.append(b)
                continue
            # left_childs = torch.stack(b.left_childs)
            left_childs = b.left_childs
            num_score_constraints=[]
            '''
            num_score_constraints.append([1]*(num_size+len(generate_nums)) )#B*(num+gen)
            if t>0:
                if t>1:
                    i_op=b.out[-2]
                    i_left=b.out[-1]
                    if i_op<num_start and i_left>=num_start+len(generate_nums):
                        idx_left=i_left-num_start-len(generate_nums)
                        #print(idx_left)
                        if unit_list_batch[idx_left]!="":
                            if 2 in cons_mode:
                                if i_op == add or i_op==sub:
                                    for idx_right in range(len(unit_list_batch)):
                                        if idx_right!=idx_left and unit_list_batch[idx_right]!="" and unit_list_batch[idx_left]!=unit_list_batch[idx_right]:
                                            num_score_constraints[0][idx_right+len(generate_nums)]=0
                            if 1 in cons_mode:
                                if i_op==mul or i_op==exp:
                                    for idx_right in range(len(unit_list_batch)):
                                        if idx_right!=idx_left and unit_list_batch[idx_right]!="" and unit_list_batch[idx_left]==unit_list_batch[idx_right]:
                                            num_score_constraints[0][idx_right+len(generate_nums)]=0
                            if 3 in cons_mode:
                                if i_op == add or i_op==sub:          
                                    for rule3_idx in rule3_list_batch:
                                        num_score_constraints[0][rule3_idx+len(generate_nums)]=0
                if 3 in cons_mode:
                    i_op=b.out[-1]
                    if i_op == add or i_op==sub:
                        for rule3_idx in rule3_list_batch:
                            num_score_constraints[0][rule3_idx+len(generate_nums)]=0
            num_score_constraints=torch.FloatTensor(num_score_constraints).cuda()
            '''
            num_score, op, current_embeddings, current_context, current_nums_embeddings,num_middle_score,op_middle = predict(
                b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                seq_mask, num_mask)


            # leaf = p_leaf[:, 0].unsqueeze(1)
            # repeat_dims = [1] * leaf.dim()
            # repeat_dims[1] = op.size(1)
            # leaf = leaf.repeat(*repeat_dims)
            #
            # non_leaf = p_leaf[:, 1].unsqueeze(1)
            # repeat_dims = [1] * non_leaf.dim()
            # repeat_dims[1] = num_score.size(1)
            # non_leaf = non_leaf.repeat(*repeat_dims)
            #
            # p_leaf = torch.cat((leaf, non_leaf), dim=1)
            out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

            # out_score = p_leaf * out_score

            topv, topi = out_score.topk(beam_size)

            outputs_middle_predict=torch.cat((op_middle,num_middle_score), 1)


            # is_leaf = int(topi[0])
            # if is_leaf:
            #     topv, topi = op.topk(1)
            #     out_token = int(topi[0])
            # else:
            #     topv, topi = num_score.topk(1)
            #     out_token = int(topi[0]) + num_start

            for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                current_node_stack = copy_list(b.node_stack)
                current_left_childs = []
                current_embeddings_stacks = copy_list(b.embedding_stack)
                current_out = copy.deepcopy(b.out)

                out_token = int(ti)
                current_out.append(out_token)

                temp_node_stack=[]
                A_matrix=[[0 for _ in range(t+1)] for _ in range(t+1)]
                for t_idx in range(t+1):
                    token_idx=current_out[t_idx]
                    if len(temp_node_stack)!=0:
                        parent_idx=temp_node_stack.pop()
                        A_matrix[parent_idx][t_idx]=1
                        A_matrix[t_idx][parent_idx]=1
                    if token_idx<num_start:
                        temp_node_stack.append(t_idx)
                        temp_node_stack.append(t_idx)


                if len(temp_node_stack)!=0:
                    parent_idx=temp_node_stack.pop()
                else:
                    parent_idx=0

                node = current_node_stack[0].pop()

                if out_token < num_start:
                    generate_input = torch.LongTensor([out_token])
                    if USE_CUDA:
                        generate_input = generate_input.cuda()
                    left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context,outputs_middle_predict)

                    current_node_stack[0].append(TreeNode(right_child))
                    current_node_stack[0].append(TreeNode(left_child, left_flag=True))
                    current_embeddings_stacks[0].append(node_label[0].unsqueeze(0))
                    #current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                else:
                    current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)
                    current_embeddings_stacks[0].append(current_num)#t*[1*H]
                    #while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                    #    sub_stree = current_embeddings_stacks[0].pop()
                    #    op = current_embeddings_stacks[0].pop()
                    #    current_num = merge(op.embedding, sub_stree.embedding, current_num)
                    #current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                
                tree_embed_mat=copy_list(current_embeddings_stacks[0])#S*H
                new_tree_embed_mat=merge(torch.stack(tree_embed_mat, dim=1),torch.FloatTensor(A_matrix).cuda())#t*1*H,t*t
                tree_embed_list=new_tree_embed_mat.split(1)#S*[1,H]
                current_left_childs.append(tree_embed_list[parent_idx])

                for t_idx in range(t+1):
                    current_embeddings_stacks[0][t_idx]=tree_embed_list[t_idx]

                #if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                #    current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                #else:
                #current_left_childs.append(None)
                current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, current_embeddings_stacks,
                                              current_left_childs, current_out))
        beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
        beams = beams[:beam_size]
        flag = True
        for b in beams:
            if len(b.node_stack[0]) != 0:
                flag = False
        if flag:
            break

    return beams[0].out
# coding: utf-8
from pre_data import *
import time
from expressions_transfer import *
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

batch_size = 64
embedding_size = 128
hidden_size = 512
n_epochs = 80
learning_rate = 1e-3
weight_decay = 1e-5
beam_size = 5
n_layers = 2
x=[1,2,3,4,5,6,7]
if 0 in x:
    print("0")
if 1 in x:
    print("1")
if 2 in x:
    print("2")
if 3 in x:
    print("3")
#data [id,ques,quessplit,equation,answer]
data = load_raw_data("data/Math_23K.json")

#pairs (input_seq, out_seq, nums, num_pos) 替换后的输入序列，输出序列，替换数字按N1N2序，替换数字的位置
#generate_nums [u'1', u'3.14']
#copy_nums 一句话里最多数字数
#输入里用NUM 替换数字，输出里用N1,N2替换数字
pairs, generate_nums, copy_nums = transfer_num(data)
print(pairs[0])
print(generate_nums)
print(copy_nums)

#把输出序列换成前序
temp_pairs = []
for p in pairs:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
pairs = temp_pairs


fold_size = int(len(pairs) * 0.2)
fold_pairs = []
for split_fold in range(4):
    fold_start = fold_size * split_fold
    fold_end = fold_size * (split_fold + 1)
    fold_pairs.append(pairs[fold_start:fold_end])
fold_pairs.append(pairs[(fold_size * 4):])

best_acc_fold = []

for fold in range(1):
    pairs_tested = []
    pairs_trained = []
    for fold_t in range(5):
        if fold_t == fold:
            pairs_tested += fold_pairs[fold_t]
        else:
            pairs_trained += fold_pairs[fold_t]

    #train_pair : input_cell, len(input_cell), output_cell, len(output_cell),pair[2], pair[3], num_stack
    #input_cell 转换成数字的序列
    #pair[2] 题中数字， pair[3] 题中数字的位置
    #num_stack 如果答案中有没替换掉的数字，如果这个数字在题目中，把这个数字在文中是第几个数字，即nums的编号记下来，如果不在题目中，把整个nums的所有编导都记下来
    #num_stack还要逆转，可能是为了逆序
    input_lang, output_lang, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_tested, 5, generate_nums,
                                                                    copy_nums, tree=True)

    print(train_pairs[0])
    print(input_lang.n_words)
    print(output_lang.n_words)

    for epoch in range(1):
        #输入，长度，输出，长度，题中数字，答案中没替换的数字，题中数字在题中的位置，题中数字量
        input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches,input_edge_batches = prepare_train_batch(train_pairs, batch_size)

        print("********************")
        print(input_edge_batches[0][0])
x=[1,2,3,4,5,6,7]
y=[0]+x
print(y)
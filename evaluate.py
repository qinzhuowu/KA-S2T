# coding: utf-8
from train_and_evaluate import *
from models import *
import time
import torch.optim
from expressions_transfer import *
from torch.optim import lr_scheduler

batch_size = 64
embedding_size = 128
hidden_size = 512
n_epochs = 80
learning_rate = 1e-3
weight_decay = 1e-5
beam_size = 5
n_layers = 2

data = load_raw_data("data/Math_23K.json")

pairs, generate_nums, copy_nums = transfer_num(data)

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

for fold in range(5):
    pairs_tested = []
    pairs_trained = []
    for fold_t in range(5):
        if fold_t == fold:
            pairs_tested += fold_pairs[fold_t]
        else:
            pairs_trained += fold_pairs[fold_t]

    input_lang, output_lang, train_pairs, test_pairs,category_vocab,hownet_dict_vocab = prepare_data(pairs_trained, pairs_tested, 5, generate_nums,
                                                                    copy_nums, tree=True)
    # Initialize models
    encoder = EncoderSeq(input_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
                         n_layers=n_layers,category_size=len(category_vocab))
    predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                         input_size=len(generate_nums))
    generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),output_vocab_len=output_lang.n_words,
                            embedding_size=embedding_size)
    merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)
    # the embedding layer is  only for generated number embeddings, operators, and paddings

    encoder.load_state_dict(torch.load("models/encoder"+str(fold)))
    predict.load_state_dict(torch.load("models/predict"+str(fold)))
    generate.load_state_dict(torch.load("models/generate"+str(fold)))
    merge.load_state_dict(torch.load("models/merge"+str(fold)))

    # Move models to GPU
    if USE_CUDA:
        encoder.cuda()
        predict.cuda()
        generate.cuda()
        merge.cuda()

    generate_num_ids = []
    for num in generate_nums:
        generate_num_ids.append(output_lang.word2index[num])

    print("fold:", fold + 1)
    #print("epoch:", epoch + 1)
    print("--------------------------------")
    out_filename="output/evaluate_test_result"+str(fold)
    out_filename1="output/evaluate_test_wrong"+str(fold)
    file_out=open(out_filename,"w")
    file_wrong=open(out_filename1,"w") 
    value_ac = 0
    equation_ac = 0
    eval_total = 0
    start = time.time()
    for test_batch in test_pairs:
        test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
                                         merge, input_lang,output_lang, test_batch[5],test_batch[7],test_batch[8],test_batch[9],
                                         test_batch[10],hownet_dict_vocab,beam_size=beam_size)
                
        val_ac, equ_ac, test_list, tar_list = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
        file_out.write(" ".join([str(x) for x in test_list])+"###"+" ".join([str(x) for x in tar_list])+"###"+" ".join(indexes_to_sentence(input_lang,test_batch[0]))+"\n")
        if val_ac:
            value_ac += 1
        else:
            file_wrong.write(" ".join([str(x) for x in test_list])+"###"+" ".join([str(x) for x in tar_list])+"###"+" ".join(indexes_to_sentence(input_lang,test_batch[0]))+"\n")
        if equ_ac:
            equation_ac += 1
        eval_total += 1
    print(equation_ac, value_ac, eval_total)
    print("test_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
    print("testing time", time_since(time.time() - start))
    print("------------------------------------------------------")
    best_acc_fold.append((equation_ac, value_ac, eval_total))
    a, b, c = 0, 0, 0
    for bl in range(len(best_acc_fold)):
        print(best_acc_fold[bl][0] / float(best_acc_fold[bl][2]), best_acc_fold[bl][1] / float(best_acc_fold[bl][2]))
        a += best_acc_fold[bl][0]
        b += best_acc_fold[bl][1]
        c += best_acc_fold[bl][2]
    print(a / float(c), b / float(c))


a, b, c = 0, 0, 0
for bl in range(len(best_acc_fold)):
    a += best_acc_fold[bl][0]
    b += best_acc_fold[bl][1]
    c += best_acc_fold[bl][2]
    print(best_acc_fold[bl])
print(a / float(c), b / float(c))

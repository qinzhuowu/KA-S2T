#coding:utf-8
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
from tensorflow.contrib.seq2seq import CustomHelper,dynamic_decode,TrainingHelper,GreedyEmbeddingHelper,BasicDecoder
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple,DropoutWrapper
import sys

from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from tensorflow.python.layers import core as layers_core
import codecs
import collections
import random

from model import Seq2seq_Attention
#from cilin import replace_same_category
from cilin_edge import read_cilin_edge,pad_node_edge,get_node_edge_from_dict
from pre_data import *
import time
from expressions_transfer import *

batch_size = 64
embedding_size = 128
hidden_size = 512
n_epochs = 80
learning_rate = 1e-3
weight_decay = 1e-5
beam_size = 5
n_layers = 2

reload(sys)
sys.setdefaultencoding( "utf-8" )

slim = tf.contrib.slim
#如果想让编码题目和解码答案用相同的词表，词表存在Data//mix_vocab里
train_encode_vec = '..//..//GCN_Data//jieba//train_recut.enc'  
train_decode_vec = '..//..//GCN_Data//train_tree.dec'  
valid_encode_vec = '..//..//GCN_Data//jieba//valid_recut.enc'  
valid_decode_vec = '..//..//GCN_Data//valid_tree.dec'
test_encode_vec = '..//..//GCN_Data//jieba//test_recut.enc'  
test_decode_vec = '..//..//GCN_Data//test_tree.dec'
encode_vocab_file="..//..//GCN_Data//jieba//encode_vocabulary_recut"
decode_vocab_file="..//..//GCN_Data//jieba//decode_vocabulary_recut"

train_number_file='..//..//GCN_Data//train_n1_number'  
valid_number_file='..//..//GCN_Data//valid_n1_number'  
test_number_file='..//..//GCN_Data//test_n1_number' 

#train_unit_file='..//..//Data//train_unit'  
#valid_unit_file='..//..//Data//valid_unit'  
#test_unit_file='..//..//Data//test_unit'  

entity_vocab_file="..//..//GCN_Data//jieba//entity_vocabulary"
entity_vocab_size=2622

T_MAX_LENGTH=100
X_VOCAB_SIZE = 4000 
Y_VOCAB_SIZE=26

X_MAX_LENGTH=50
Y_MAX_LENGTH=30


PAD_ID = 0 
GO_ID = 0  #这难道就是所谓的SOS ID?
EOS_ID = 0  #问题最后表示结束
UNK_ID=3

embedding_size = 200 
num_trans_units=100
Y_embedding_size=200
hidden_size = 512  # 每层大小  not 256 is 512
n_layers = 2     # 层数 
encoder_layers=2
decoder_layers=2
batch_size = 64  #批次
#slot_size = 122
#intent_size = 22
epoch_num = 200
max_epoch_num=epoch_num*3
inference=False

learning_rate = 0.001
learning_rate_decay = 0.1

state_size = 128  #64, 128, 256 was a good number for linux OS!
logs_path = 'tmp/logs'
Embeddingfilename = '..//..//Data//sgns.wiki.bigram'

tf.flags.DEFINE_boolean("test",False,'Test?')

FLAGS=tf.flags.FLAGS

def get_vocab():
    x_idx_to_word=[]
    y_idx_to_word=[]
    i=0
    count=0
    prcentage=0.0
    encode_vocab_dataset = codecs.open(encode_vocab_file, "r", encoding="UTF-8").readlines()
    decode_vocab_dataset = codecs.open(decode_vocab_file, "r", encoding="UTF-8").readlines()
    a1_list=['A0', 'A1', 'A2', 'A3', 'A4', 'B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'C0', 'C1', 'C2', 'C3', 'C4', 'D0', 'D1', 'D2', 'D3', 'D4', 'E0', 'E1', 'E2', 'E3', 'F0', 'F1', 'F2', 'G0', 'G1', 'H0', 'H1', 'H2', 'I0', 'I1', 'I2', 'I3']
    for line in encode_vocab_dataset:
        x_idx_to_word.append(line.strip())
    x_idx_to_word1=x_idx_to_word[:Y_VOCAB_SIZE]
    x_idx_to_word2=x_idx_to_word[Y_VOCAB_SIZE:]
    x_idx_to_word=x_idx_to_word1+a1_list+x_idx_to_word2
    x_idx_to_word=x_idx_to_word[:X_VOCAB_SIZE]
    for line in decode_vocab_dataset:
        y_idx_to_word.append(line.strip())

    x_word_to_idx = {word:ix for ix, word in enumerate(x_idx_to_word)}
    y_word_to_idx = {word:ix for ix, word in enumerate(y_idx_to_word)}
    return x_idx_to_word, x_word_to_idx, y_idx_to_word, y_word_to_idx

x_idx_to_word, x_word_to_idx, y_idx_to_word, y_word_to_idx=get_vocab()

print(len(y_idx_to_word))


def loadGloVe(filename,vocab):
    embd = []
    file = open(filename,'r')
    dictvocab={}
    for line in file.readlines():
        row = line.strip().split(' ')
        row_digits=[]
        for digits in range(1,len(row)):
            row_digits.append(float(row[digits]))
        dictvocab[row[0]]=row_digits
    for vocab_word in vocab:
        embd.append(dictvocab.get(vocab_word,[0.0]*embedding_size))
    print('Loaded GloVe!')
    print(len(embd))
    file.close()
    return embd



#embd = loadGloVe(Embeddingfilename,x_idx_to_word)
#embedding = np.asarray(embd)


# 读取*dencode.vec和*decode.vec数据（数据还不算太多, 一次读人到内存）  

def generate_text(prediction, batch_size, length, vocab_size, idx_to_word):

    #batch_softmax = np.reshape(prediction, [batch_size, length, vocab_size])
    batch_sentence = []
    #print(len(prediction))

    for sequence in prediction:
        #print(len(sequence))
        word_sequence = ''
        for char in sequence:
            #vector_position = np.argmax(char)
            y_word = idx_to_word[char]
            if y_word != 'ZERO' and y_word!="__PAD__":
                word_sequence = word_sequence + y_word + ' '
            else:
                word_sequence = word_sequence + ''
        batch_sentence.append(word_sequence)

    return batch_sentence

def que2ids(sentences):
    ids = []
    nm_word = []
    nm_id = []
    for id_ in range(len(sentences)):
        w=sentences[id_]
        _id = x_word_to_idx.get(w,UNK_ID)
        ids.append(_id)

    for id_ in range(len(sentences)):
        list_temp=[PAD_ID,PAD_ID,PAD_ID]
        ids1=list_temp+ids[:]+list_temp
        w=sentences[id_]
        if w in y_idx_to_word:
            if w.startswith("n") or w.startswith("m"):
                if w not in nm_word:
                    nm_word.append(w)
                    ids2=ids1[id_:id_+7]
                    nm_id.append(ids2)

    return ids,nm_word,nm_id

def get_one_hot_vector():
    pad_vector=[]
    add_vector=[]
    multi_vector=[]
    number_vector=[]
    add_list=["+","-"]
    multi_list=["*","/","^"]
    for word in y_idx_to_word:
        pad_vector.append(0)
        if word in add_list:
            add_vector.append(1)
        else:
            add_vector.append(0)
        if word in multi_list:
            multi_vector.append(1)
        else:
            multi_vector.append(0)
        if word.startswith("n") or word.startswith("m"):
            number_vector.append(1)
        else:
            number_vector.append(0)
    return pad_vector,add_vector,multi_vector,number_vector

operator_unit = {
    '+' : 1,
    '-' : 2,
    '*' : 3,
    '/' : 4,
    '^' : 5
}

def ans2ids_simple(sentences):
    ids = []
    for w in sentences:
        _id = y_word_to_idx.get(w,UNK_ID)
        ids.append(_id)
    return ids


def ans2ids(sentences,nm_word):
    ids = []
    for w in sentences:
        _id = y_word_to_idx.get(w,UNK_ID)
        ids.append(_id)
    tree_list=sentences[:]

    i=0
    while i< len(tree_list):
        if tree_list[i] not in operator_unit:
            if tree_list[i] not in nm_word:
                tree_list[i]=0
        i=i+1
    pad_str=[]
    max_len=len(nm_word)    
    temp_pad_list=[]
    for j in range(max_len):
        temp_pad_list.append(0)
    for i in range(max_len):
        pad_str.append(temp_pad_list)
    i=0
    while i< len(tree_list):
        if tree_list[i] in operator_unit:
            if tree_list[i-1] !=0 and tree_list[i-2] !=0 :
                for j in tree_list[i-2].split():
                    for k in tree_list[i-1].split():
                        if j in nm_word and k in nm_word:
                            index1=nm_word.index(j)
                            index2=nm_word.index(k)
                            pad_str[index1][index2]=operator_unit[tree_list[i]]
                            pad_str[index2][index1]=operator_unit[tree_list[i]]
                tree_list[i-2]= str(tree_list[i-2])+" "+str(tree_list[i-1])
                del(tree_list[i])
                del(tree_list[i-1])
                i=0
            elif tree_list[i-1] ==0:
                del(tree_list[i])
                del(tree_list[i-1])
                i=0
            elif tree_list[i-2] ==0:
                tree_list[i-2]= str(tree_list[i-1])
                del(tree_list[i])
                del(tree_list[i-1])
                i=0
        else:
            i=i+1

    return ids,pad_str






class BatchManager(object):
    """
    Mini-Batch Manager Class with padding.
    """
    def __init__(self,data,vocab_oovs=False,extend_vocab=False,shuffle=False):
        self.data=data
        self.batch_size=batch_size
        self.num_batch=len(data)//batch_size
        self.vocab_X = x_word_to_idx
        self.vocab_Y=y_word_to_idx
        self.pad_id = PAD_ID
        self.vocab_oovs = vocab_oovs
        self.extend_vocab = extend_vocab
        self.shuffle = shuffle

    def _prepare_data(self,data):
        #src_dataset,pre_dataset,tgt_dataset = zip(*data)
        src_ids_dataset = []
        #pre_ids_dataset = []
        tgt_ids_dataset = []

        src_node_dataset=[]
        src_edge_dataset=[]
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        num_size_batch = []
        for i, li, j, lj, num, num_pos, num_stack in data:
            src_ids_dataset.append(i)
            tgt_ids_dataset.append(j)
            num_batch.append(num)
            num_stack_batch.append(num_stack)
            num_pos_batch.append(num_pos)
            num_size_batch.append(len(num_pos))


        return src_ids_dataset, tgt_ids_dataset,num_batch,num_stack_batch,num_pos_batch,num_size_batch

    def _repeat_data(self, datas):
        batch_size = self.batch_size
        _data = []
        for data in datas:
            for i in range(batch_size):
                _data.append(data)
        return _data

    def _pad_src_data(self, src_ids_dataset, max_len):
        pad_src_batch = []
        if max_len!=None:
            max_src_length = max_len
        else:
            max_src_length = max([len(src) for src in src_ids_dataset])
        src_length = []
        src_padding_mask = np.zeros((self.batch_size, max_src_length), dtype = np.float32)
        for i in range(len(src_ids_dataset)):
            src_id  = src_ids_dataset[i]
            if len(src_id)>max_src_length:
                src_id=src_id[-max_src_length:]
            paddings = [self.pad_id]*(max_src_length - len(src_id))
            pad_src_batch.append(src_id+paddings)
            src_length.append(len(src_id))
            for j in range(len(src_id)):
                src_padding_mask[i][j] = 1
        return pad_src_batch, src_length, src_padding_mask,max_src_length
    def _pad_src_data_original(self, src_ids_dataset, max_len):
        pad_src_batch = []
        if max_len!=None:
            max_src_length = max_len
        else:
            max_src_length = max([len(src) for src in src_ids_dataset])
        src_length = []
        src_padding_mask = np.zeros((self.batch_size, max_src_length), dtype = np.float32)
        for i in range(len(src_ids_dataset)):
            src_id  = src_ids_dataset[i]
            paddings = [self.pad_id]*(max_src_length - len(src_id))
            pad_src_batch.append(src_id+paddings)
            src_length.append(len(src_id))
            for j in range(len(src_id)):
                src_padding_mask[i][j] = 1
        return pad_src_batch, src_length, src_padding_mask,max_src_length
    def _pad_tgt_data(self, tgt_ids_dataset, max_len):
        sos_id = GO_ID
        eos_id = EOS_ID
        inps = []
        targets = []
        for tgt_ids in tgt_ids_dataset:
            inp = [sos_id]+tgt_ids[:]
            target = tgt_ids[:] + [eos_id]
            if len(inp) > max_len:
                inp = inp[:max_len]
                target = target[:max_len]
            inps.append(inp)
            targets.append(target)
        pad_tgt_in_batch, tgt_length, tgt_padding_mask,_ = self._pad_src_data_original(inps, max_len)
        pad_tgt_out_batch, _, _,_ = self._pad_src_data_original(targets, max_len)
        return pad_tgt_in_batch, pad_tgt_out_batch, tgt_length, tgt_padding_mask
    '''
    def _pad_number_list(self, unit_n1n2_dataset, unit_oper_dataset):
        pad_vector,add_vector,multi_vector,number_vector=get_one_hot_vector()
        pad_number_batch = []
        max_src_length = max([len(src) for src in unit_n1n2_dataset])

        #list_temp_pad=[0]*4
        number_length = []
        pad_number_oper_batch=[]
        for i in range(len(unit_n1n2_dataset)):
            src_id  = unit_n1n2_dataset[i]
            need_to_pad=max_src_length - len(src_id)
            paddings=[]
            for j in range(need_to_pad):
                list_temp_pad=[i,max_src_length-1,i,max_src_length-1]
                paddings.append(list_temp_pad)
            pad_number_batch.append(src_id+paddings)
            number_length.append(len(src_id))

        for i in range(len(unit_n1n2_dataset)):
            src_id  = unit_oper_dataset[i]
            need_to_pad=max_src_length - len(src_id)
            paddings=[]
            for j in range(need_to_pad):
                paddings.append(pad_vector)
            pad_number_oper_batch.append(src_id+paddings)

        if len(pad_number_batch)!=batch_size:
            print(len(pad_number_batch))
        for i in range(len(pad_number_batch)):
            if len(pad_number_batch[i])!=max_src_length:
                print(len(pad_number_batch[i]))
            for j in range(len(pad_number_batch[i])):
                if len(pad_number_batch[i][j])!=4:
                    print(len(pad_number_batch[i][j]))

        if len(pad_number_oper_batch)!=batch_size:
            print(len(pad_number_oper_batch))
        for i in range(len(pad_number_oper_batch)):
            if len(pad_number_oper_batch[i])!=max_src_length:
                print(len(pad_number_oper_batch[i]))
            for j in range(len(pad_number_oper_batch[i])):
                if len(pad_number_oper_batch[i][j])!=Y_VOCAB_SIZE:
                    print(len(pad_number_oper_batch[i][j]))

        return pad_number_batch, number_length, pad_number_oper_batch
    '''
    def _pad_data(self,data):
        #src_ids_dataset, pre_ids_dataset, tgt_ids_dataset = self._prepare_data(data)
        src_ids_dataset, tgt_ids_dataset,num_batch,num_stack_batch,num_pos_batch,num_size_batch = self._prepare_data(data)
        src_oovs = None
        max_src_oovs = None
        #print(src_ids_dataset[0])
        #print(tgt_ids_dataset[0])

        pad_src_batch, src_length, src_padding_mask,max_seq_length = self._pad_src_data(src_ids_dataset, None)
        #pad_pre_batch, pre_length, pre_padding_mask = self._pad_src_data(src_ids_dataset, None)
        pad_tgt_in_batch, pad_tgt_out_batch, tgt_length, tgt_padding_mask = self._pad_tgt_data(tgt_ids_dataset, None)
        #print(pad_tgt_in_batch[0])
        #print(pad_tgt_out_batch[0])
        return BatchedInput(
            src_batch=pad_src_batch,
            src_length=src_length,
            src_padding_mask=src_padding_mask,
            num_batch=num_batch,
            num_stack_batch=num_stack_batch,
            num_pos_batch=num_pos_batch,
            num_size_batch=num_size_batch,
            #pad_node_batch=pad_node_batch,
            #pad_edge_batch=pad_edge_batch,
            #pre_batch=pad_pre_batch,
            #pre_length=pre_length,
            #pre_padding_mask=pre_padding_mask,
            #pad_number_batch=pad_number_batch, 
            #number_length=number_length, 
            #pad_number_oper_batch=pad_number_oper_batch,
            tgt_in_batch=pad_tgt_in_batch,
            tgt_out_batch=pad_tgt_out_batch,
            tgt_length=tgt_length,
            tgt_padding_mask=tgt_padding_mask,
            src_oovs=src_oovs,
            max_src_oovs=max_src_oovs)

    def iter_batch(self):
        if self.shuffle:
            random.shuffle(self.data)
        for i in range(self.num_batch):
            batch_data=self.data[i*self.batch_size:(i+1)*self.batch_size]
            yield self._pad_data(batch_data)
        '''
        if not self.shuffle:
            batch_data=self.data[self.num_batch*self.batch_size:]
            yield self._pad_data(batch_data)
        '''
class BatchedInput(collections.namedtuple("BatchedInput",
                                          ("src_batch","src_length","src_padding_mask",
                                            "num_batch","num_stack_batch","num_pos_batch","num_size_batch",
                                            #"pad_node_batch","pad_edge_batch",
                                            #"pad_number_batch", "number_length", "pad_number_oper_batch",
                                        #  "pre_batch","pre_length","pre_padding_mask",
                                          "tgt_in_batch","tgt_out_batch","tgt_length","tgt_padding_mask","src_oovs","max_src_oovs"))):
  pass
def vectorize_data(word_sentences, max_length, word_to_idx):
    sequences = np.zeros((len(word_sentences), max_length, len(word_to_idx)), dtype=float)
    for i, sentence in enumerate(word_sentences):
        for j, word in enumerate(sentence):
            sequences[i, j, word] = 1.
    return sequences


#encoder_x = tf.placeholder(dtype=tf.int32, shape=[None, None]) #[batch_size, X_MAX_LENGTH]
#decoder_x = tf.placeholder(dtype=tf.int32, shape=[None, None, Y_VOCAB_SIZE]) #[batch_size, Y_MAX_LENGTH, Y_VOCAB_SIZE]
#y = tf.placeholder(dtype=tf.float32, shape=[None, None, Y_VOCAB_SIZE])#[batch_size, Y_MAX_LENGTH, Y_VOCAB_SIZE]
#init_state = tf.placeholder(tf.float32, [n_layers, 2, batch_size, state_size])

#embedding_placeholder = tf.placeholder(tf.float32, [X_VOCAB_SIZE, embedding_size])
#encoder_length=tf.placeholder(dtype=tf.int32, shape=[batch_size])
#triples_x=tf.placeholder(dtype=tf.int32, shape=[None, X_MAX_LENGTH,T_MAX_LENGTH,3]) #[batch_size, X_MAX_LENGTH]


def loadEntityVocab(filename,vocab):
    entity_vocab = []
    entity_vocab_dataset = codecs.open(filename, "r", encoding="UTF-8").readlines()
    for line in entity_vocab_dataset:
        _id = x_word_to_idx.get(line.strip(),UNK_ID)
        if _id==UNK_ID:
            _id==UNK_ID
            #print("wrong")
            #print(line.strip())
        else:
            entity_vocab.append(_id)
    return entity_vocab

entity_vocab = loadEntityVocab(entity_vocab_file,x_idx_to_word)
entity_vocab = np.asarray(entity_vocab)

cilin_edge,all_word_list=read_cilin_edge()
cilin_edge= np.asarray(cilin_edge)
#cilin_edge =cilin_edge.astype(np.float32)
I = np.matrix(np.eye(cilin_edge.shape[0]))
#cilin_edge=cilin_edge+I
cilin_edge=tf.cast(cilin_edge,dtype=tf.float32)



def get_batch_raw(data_set, bucket_id,batch_num):
    start_batch = bucket_id * batch_num
    end_batch = start_batch + batch_size
    return data_set[start_batch:end_batch]



def get_correct_answer():
    n1_to_number=[]
    correct_answer_list=[]
    check_question_list=[]

    train_correct_answer=[]
    f= codecs.open(train_number_file, "r", encoding="UTF-8").readlines()
    for line in f:
        list_data=line.strip().split("###")
        train_correct_answer.append(list_data[3])
        #print(list_data[1])

    valid_correct_answer=[]
    f= codecs.open(valid_number_file, "r", encoding="UTF-8").readlines()
    for line in f:
        list_data=line.strip().split("###")
        valid_correct_answer.append(list_data[3])

    test_correct_answer=[]
    f= codecs.open(test_number_file, "r", encoding="UTF-8").readlines()
    for line in f:
        list_data=line.strip().split("###")
        test_correct_answer.append(list_data[3])
    return train_correct_answer,valid_correct_answer,test_correct_answer
train_correct_answer,valid_correct_answer,test_correct_answer=get_correct_answer()



all_word=["+","-","*","/","(",")","^","[","]","1","2","3","4","5","6","7","8","9","0","n","p","m","%"," ",".","*",":"]
operator_unit = {
    '+' : 1,
    '-' : 2,
    '*' : 3,
    '/' : 4,
    '^' : 5
}
def oper(x1,x2,operator,flag):
    if flag==0:
        if operator=="+":
            x0=float(x1)+float(x2)
        elif operator=="-":
            x0=float(x1)-float(x2)
        elif operator=="*":
            x0=float(x1)*float(x2)
        elif operator=="/":
            if float(x2)==0:
                x0=float(x1)
            else:
                x0=float(x1)/float(x2)
        elif operator=="^":
            print(str(x1)+" ^ "+str(x2))
            if x1> 10000 or x2>20:
                x0=float(x1)
            else:
                x0=float(x1)**float(x2)
        return x0
    if flag==1:
        return x1
def get_answer(gen_exp,corr_exp,number_file,correct_answer_list):
    f= codecs.open(number_file, "r", encoding="UTF-8").readlines()
    n1_to_number=[]
    m1_to_number=[]
    id_=0
    for line in f:
        #print(line)
        n1_to_number.append(line.split("###")[6])
        #print(line.strip("###")[6])
        #if id_ %4 == 3:
        #    m1_to_number.append(line.strip())
        id_+=1
    gen_ans=[]
    corr_ans=[]
    ground_ans=[]
    check_ans=[]
    for id_ in range(len(gen_exp)):
        temp_str_number=correct_answer_list[id_]
        if temp_str_number.endswith("%"):
            temp_str_number=temp_str_number.replace("%","")
            temp_str_number=float(temp_str_number)*0.01
        elif '/' in temp_str_number:
            if '((' in temp_str_number:
                for j in range(0,len(temp_str_number)-1):
                    if temp_str_number[j]>='0' and temp_str_number[j]<='9': 
                        if temp_str_number[j+1]=='(':
                            temp_str_number=temp_str_number[:j+1]+"+"+temp_str_number[j+1:]
            temp_str_number=temp_str_number.replace("(","")
            temp_str_number=temp_str_number.replace(")","")
            if "+" in temp_str_number:
                temp_list1=temp_str_number.split("+")
                temp_list2=temp_list1[1].split("/")
                temp_str_number=float(temp_list1[0])+(float(temp_list2[0])/float(temp_list2[1]))
            else:
                temp_list=temp_str_number.split("/")
                #print(temp_str_number)
                temp_str_number=float(temp_list[0])/float(temp_list[1])
        
        ground_ans.append(float(temp_str_number))

        temp_str_n=n1_to_number[id_]
        check_ans.append(temp_str_n)
        #temp_str_m=m1_to_number[id_]


        for char_temp in temp_str_n:
            if char_temp not in all_word:
                temp_str_n=temp_str_n.replace(char_temp,"")

        n1_number_temp={}
        #m1_number_temp={}
        #print(temp_str_n)
        list_n1_number=temp_str_n.split("*")[:-1]
        #list_m1_number=temp_str_m.split(":")[1].split()
        for temp_word in list_n1_number:
            if len(temp_word) > 1:
                temp_str=temp_word.split(" ")[1]
                temp_n1=temp_word.split(" ")[0]
                if '/' in temp_str:
                    temp_str=temp_str.replace("(","")
                    temp_str=temp_str.replace(")","")
                    if "+" in temp_str:
                        temp_list1=temp_str.split("+")
                        temp_list2=temp_list1[1].split("/")
                        n1_number_temp[temp_n1]=float(temp_list1[0])+(float(temp_list2[0])/float(temp_list2[1]))
                        print("get here")
                    else:
                        temp_list=temp_str.split("/")
                        n1_number_temp[temp_n1]=float(temp_list[0])/float(temp_list[1])
                elif temp_str.endswith("%"):
                    temp_str=temp_str.replace("%","")
                    n1_number_temp[temp_n1]=float(temp_str)*0.01
                else:
                    n1_number_temp[temp_n1]=float(temp_str)
        '''
        for temp_word in list_m1_number:
            if len(temp_word) > 1:
                temp_str=temp_word.split("*")[0]
                temp_n1=temp_word.split("*")[1]
                if '/' in temp_str:
                    temp_str=temp_str.replace("(","")
                    temp_str=temp_str.replace(")","")
                    if "+" in temp_str:
                        temp_list1=temp_str.split("+")
                        temp_list2=temp_list1[1].split("/")
                        m1_number_temp[temp_n1]=float(temp_list1[0])+(float(temp_list2[0])/float(temp_list2[1]))
                        print("get here")
                    else:
                        temp_list=temp_str.split("/")
                        m1_number_temp[temp_n1]=float(temp_list[0])/float(temp_list[1])
                elif temp_str.endswith("%"):
                    temp_str=temp_str.replace("%","")
                    m1_number_temp[temp_n1]=float(temp_str)*0.01
                else:
                    m1_number_temp[temp_n1]=float(temp_str)
        '''
        not_used_number=[]
        for temp_word in n1_number_temp:
            if n1_number_temp[temp_word] not in not_used_number:
                not_used_number.append(n1_number_temp[temp_word])
        '''
        for temp_word in m1_number_temp:
            if m1_number_temp[temp_word] not in not_used_number:
                not_used_number.append(m1_number_temp[temp_word])
        '''
        list_gen=gen_exp[id_].split()
        i=0
        while i < len(list_gen):
            if list_gen[i] in n1_number_temp:
                list_gen[i] = n1_number_temp.get(list_gen[i])
                if list_gen[i] in not_used_number:
                    not_used_number.remove(list_gen[i])
            elif list_gen[i] == "p":
                list_gen[i] = 3.14
            else:
                if str(list_gen[i]).startswith("m") or str(list_gen[i]).startswith("n"):
                    if len(not_used_number) !=0:
                        list_gen[i]=float(not_used_number[0])
                        del(not_used_number[0])
                    else:
                        list_gen[i]=1.0

            if list_gen[i] in operator_unit:
                if i<2:
                    break
                elif str(list_gen[i-2]).startswith("_") or str(list_gen[i-2]).startswith("n"):
                    del(list_gen[i])
                    del(list_gen[i-2])
                    i=0
                elif str(list_gen[i-1]).startswith("_") or str(list_gen[i-1]).startswith("n"):
                    del(list_gen[i])
                    del(list_gen[i-1])
                    i=0
                else:
                    list_gen[i-2]=oper(list_gen[i-2],list_gen[i-1],list_gen[i],0)
                    del(list_gen[i])
                    del(list_gen[i-1])
                    i=0
            else:
                i=i+1
        
        if len(list_gen)!=0:
            if list_gen[0] not in operator_unit:
                gen_ans.append(list_gen[0])
            else:
                gen_ans.append(0)
        else:
            gen_ans.append(0)
        
        list_gen=corr_exp[id_].split()
        i=0
        while i < len(list_gen):
            if list_gen[i] in n1_number_temp:
                list_gen[i] = n1_number_temp.get(list_gen[i])
            elif list_gen[i] == "p":
                list_gen[i] = 3.14

            if list_gen[i] in operator_unit:
                if i<2:
                    break
                elif str(list_gen[i-2]).startswith("_") or str(list_gen[i-2]).startswith("n"):
                    del(list_gen[i])
                    del(list_gen[i-2])
                    i=0
                elif str(list_gen[i-1]).startswith("_") or str(list_gen[i-1]).startswith("n"):
                    del(list_gen[i])
                    del(list_gen[i-1])
                    i=0
                else:
                    list_gen[i-2]=oper(list_gen[i-2],list_gen[i-1],list_gen[i],0)
                    del(list_gen[i])
                    del(list_gen[i-1])
                    i=0
            else:
                i=i+1
        corr_ans.append(list_gen[0])

    return gen_ans,corr_ans,ground_ans,check_ans

def out_expression_list(test, output_lang, num_list, num_stack=None):
    max_index = output_lang.n_words
    res = []
    for i in test:
        # if i == 0:
        #     return res
        if i < max_index - 1:
            idx = output_lang.index2word[i]
            if idx[0] == "N":
                if int(idx[1:]) >= len(num_list):
                    return None
                res.append(num_list[int(idx[1:])])
            else:
                res.append(idx)
        else:
            if num_stack!=None and len(num_stack)>=1:
                pos_list = num_stack.pop()
                c = num_list[pos_list[0]]
                res.append(c)
    return res


def compute_prefix_expression(pre_fix):
    st = list()
    operators = ["+", "-", "^", "*", "/"]
    pre_fix = deepcopy(pre_fix)
    pre_fix.reverse()
    for p in pre_fix:
        if p not in operators:
            pos = re.search("\d+\(", p)
            if pos:
                st.append(eval(p[pos.start(): pos.end() - 1] + "+" + p[pos.end() - 1:]))
            elif p[-1] == "%":
                st.append(float(p[:-1]) / 100)
            else:
                st.append(eval(p))
        elif p == "+" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a + b)
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a * b)
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a * b)
        elif p == "/" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            if b == 0:
                return None
            st.append(a / b)
        elif p == "-" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a - b)
        elif p == "^" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            if float(eval(b)) != 2.0 or float(eval(b)) != 3.0:
                return None
            st.append(a ** b)
        else:
            return None
    if len(st) == 1:
        return st.pop()
    return None


def compute_prefix_tree_result(test_res, test_tar, output_lang, num_list, num_stack):
    #print(test_res, test_tar)
    #print(num_list, num_stack)

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
        if abs(compute_prefix_expression(test) - compute_prefix_expression(tar)) < 1e-4:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar

def time_since(s):  # compute time
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)

def lstm_train():

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



    #Keep previous 30 model checkpoints..
    count_all=0
    count_right=0

    with tf.Session() as sess: 
        with tf.variable_scope('training_procedure'):
            best_epoch = tf.get_variable('best_epoch',shape=[],initializer=tf.zeros_initializer(),trainable=False,dtype=tf.int32)
            best_dev_score = tf.get_variable('best_dev_score',shape=[],initializer=tf.zeros_initializer(),trainable=False,dtype=tf.float32)
        
        saver = tf.train.Saver(tf.global_variables(),max_to_keep=5)
        try:
            checkpoint = tf.train.latest_checkpoint('./TencentSeqModel')
            saver.restore(sess,checkpoint)
            print('Restore model from %s.' % checkpoint)
        except:
            sess.run(tf.global_variables_initializer())
        start_epoch,best_dev_sentence_precision = sess.run([best_epoch,best_dev_score])
        no_improve = 0
        epoch=start_epoch+1
        #Unlimited epochs for Training
        
        for fold in range(5):
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
            # (self,mode,source_vocab_size,target_vocab_size,emb_dim,encoder_num_units,encoder_num_layers,decoder_num_units, decoder_num_layers,
            #       dropout_emb,dropout_hidden,tgt_sos_id,tgt_eos_id,learning_rate,clip_norm,attention_option,beam_size,optimizer):
            op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums)
            with tf.variable_scope("root"):
                train_model=Seq2seq_Attention('train',X_VOCAB_SIZE,Y_VOCAB_SIZE,embedding_size,num_trans_units,encoder_layers,
                    num_trans_units,decoder_layers,0.5,0.5,GO_ID,EOS_ID,0.001,5,"bahdanau",5,"adam",output_lang,generate_nums,op_nums)
            with tf.variable_scope("root",reuse=True):
                dev_model=Seq2seq_Attention('infer',X_VOCAB_SIZE,Y_VOCAB_SIZE,embedding_size,num_trans_units,encoder_layers,
                    num_trans_units,decoder_layers,0,0,GO_ID,EOS_ID,0.001,5,"bahdanau",5,"adam",output_lang,generate_nums,op_nums)
                dev_sentence_precision_value = tf.placeholder(dtype=tf.float32,name='dev_sentence_precision')
                dev_sentence_precision1_summary = tf.summary.scalar(name='dev_sentence_precision',tensor=dev_sentence_precision_value)        

            train_manager = BatchManager(train_pairs,False,False, shuffle=True)
            test_manager=BatchManager(test_pairs,False,False, shuffle=False)

            for batch_data in test_manager.iter_batch():
                print(batch_data.src_batch)
                print(batch_data.tgt_in_batch)
                print(batch_data.tgt_out_batch)
                print("**********************************************")

            print(input_lang.n_words)
            print(output_lang.n_words)        
            while epoch<=max_epoch_num: 
                batch=0
                train_loss=[]
                
                start = time.time()
                for batch_data in train_manager.iter_batch():
                    batch_loss,summaries,global_step=train_model.train_step(sess,batch_data)    
                    if batch%10==0:
                        print("Epoch %d . Batch: %d" % (epoch,batch))
                    train_loss.append(batch_loss)
                    batch+=1
                print("Epoch %d finished. Loss: %.4f Global_step: %.4f" % (epoch,np.mean(train_loss),global_step))

                
                querys = []
                predicts = []
                goldens = []
                src_oovs = []
                generate_num_ids = []
                value_ac = 0
                equation_ac = 0
                eval_total = 0
                start = time.time()
                for num in generate_nums:
                    generate_num_ids.append(output_lang.word2index[num])

                
                start = time.time()
                #这里之前nums_batches不是文中所有数字的列表，而是文中有多少个数字，所以出错，现在在prepare_test_batch里改了
                #input_batches, input_lengths, output_batches, output_lengths, num_number_batches, num_stack_batches, num_pos_batches, num_size_batches = prepare_test_batch(test_pairs, batch_size)

                for batch_data in test_manager.iter_batch():
                    query_id,golden_id,predict_id=dev_model.eval_step(sess,batch_data)
                    querys.extend(query_id)
                    goldens.extend(golden_id)
                    predicts.extend(predict_id)
                print(predicts[0])
                print(goldens[0])
                    #print(input_batches[idx])
                    #print(predict_id)
                for idx in range(len(predicts)):
                    val_ac, equ_ac, _, _ = compute_prefix_tree_result(predicts[idx], test_pairs[idx][2], output_lang,test_pairs[idx][4], test_pairs[idx][6])
                    if val_ac:
                        value_ac += 1
                    if equ_ac:
                        equation_ac += 1
                    eval_total += 1
                print(equation_ac, value_ac, eval_total)
                print("test_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
                #print("testing time", time_since(time.time() - start))
                print("------------------------------------------------------")
                dev_precision=float(value_ac) / eval_total


                if dev_precision > best_dev_sentence_precision:
                    best_dev_sentence_precision = dev_precision
                    sess.run(best_epoch.assign(epoch))
                    sess.run(best_dev_score.assign(best_dev_sentence_precision))
                    save_path = saver.save(sess,"./TencentSeqModel/seq2seq.ckpt", epoch)
                    print("[STATUS] Model ckpt saved in file: %s" % save_path)
                    no_improve =0
                else:
                    no_improve = no_improve+1
                    print('\nCurrent percision {}  Best percision {}  '.format(dev_precision,best_dev_sentence_precision))
                    if no_improve >= epoch_num:
                        break
                #saver.save(sess, logs_path + '/model.ckpt', epoch)
                epoch+=1
                count_all=0
                count_right=0

#lstm_train()

def lstm_test():

    #train_set·i]里面两个列表，第一个是问题，第二个是答案
    #train_x_set,train_y_set,train_z_set,train_set_size=read_data(train_encode_vec,train_decode_vec,train_triples_file)
    #test_x_set,test_y_set,test_z_set,test_set_size=read_data(test_encode_vec,test_decode_vec,test_triples_file)
    # train_set_size 21162,test_set_size 2000
    train_src_dataset = codecs.open(train_encode_vec, "r", encoding="UTF-8").readlines()
    train_tgt_dataset = codecs.open(train_decode_vec, "r", encoding="UTF-8").readlines()
    #train_unit_dataset = codecs.open(train_unit_file, "r", encoding="UTF-8").readlines()

    #train_unit_dataset=[train_unit_dataset[id_].strip()+"#_#"+train_unit_dataset[id_+1].strip()+"#_#"+train_unit_dataset[id_+2].strip() for id_ in range(0,len(train_unit_dataset),3)]
    train_data = zip(train_src_dataset,train_tgt_dataset)

    train_manager = BatchManager(train_data,False,False, shuffle=True)
    dev_src_dataset = codecs.open(valid_encode_vec, "r", encoding="UTF-8").readlines()
    dev_tgt_dataset = codecs.open(valid_decode_vec, "r", encoding="UTF-8").readlines()
    #dev_unit_dataset = codecs.open(valid_unit_file, "r", encoding="UTF-8").readlines()
    #dev_unit_dataset=[dev_unit_dataset[id_].strip()+"#_#"+dev_unit_dataset[id_+1].strip()+"#_#"+dev_unit_dataset[id_+2].strip() for id_ in range(0,len(dev_unit_dataset),3)]
    dev_data = zip(dev_src_dataset,dev_tgt_dataset)
    dev_manager = BatchManager(dev_data,False,False,False)

    test_src_dataset = codecs.open(test_encode_vec, "r", encoding="UTF-8").readlines()
    test_tgt_dataset = codecs.open(test_decode_vec, "r", encoding="UTF-8").readlines()
    #test_unit_dataset = codecs.open(test_unit_file, "r", encoding="UTF-8").readlines()
    #test_unit_dataset=[test_unit_dataset[id_].strip()+"#_#"+test_unit_dataset[id_+1].strip()+"#_#"+test_unit_dataset[id_+2].strip() for id_ in range(0,len(test_unit_dataset),3)]
    test_data = zip(test_src_dataset, test_tgt_dataset)
    test_manager = BatchManager(test_data,False,False,False)


    #Keep previous 30 model checkpoints..
    count_all=0
    count_right=0

    with tf.Session() as sess:
        # (self,mode,source_vocab_size,target_vocab_size,emb_dim,encoder_num_units,encoder_num_layers,decoder_num_units, decoder_num_layers,
        #       dropout_emb,dropout_hidden,tgt_sos_id,tgt_eos_id,learning_rate,clip_norm,attention_option,beam_size,optimizer):
        with tf.variable_scope("root"):
            train_model=Seq2seq_Attention('train',X_VOCAB_SIZE,Y_VOCAB_SIZE,embedding_size,num_trans_units,encoder_layers,
                num_trans_units,decoder_layers,0.5,0.5,GO_ID,EOS_ID,0.001,5,"bahdanau",5,"adam")
        with tf.variable_scope("root",reuse=True):
            dev_model=Seq2seq_Attention('infer',X_VOCAB_SIZE,Y_VOCAB_SIZE,embedding_size,num_trans_units,encoder_layers,
                num_trans_units,decoder_layers,0,0,GO_ID,EOS_ID,0.001,5,"bahdanau",5,"adam")
            dev_sentence_precision_value = tf.placeholder(dtype=tf.float32,name='dev_sentence_precision')
            dev_sentence_precision1_summary = tf.summary.scalar(name='dev_sentence_precision',tensor=dev_sentence_precision_value)        

        saver = tf.train.Saver(tf.global_variables(),max_to_keep=5)
        epoch = 0
        ckpt = tf.train.get_checkpoint_state('./TencentSeqModel')
        ckptnum=0

        if ckpt != None:
            print('Loading variables from {}'.format(ckpt.model_checkpoint_path))
            epoch=int(ckpt.model_checkpoint_path.split("-")[-1])
            saver.restore(sess, ckpt.model_checkpoint_path)
            ckptnum=epoch
        else:
            sess.run(tf.global_variables_initializer())


        #Unlimited epochs for Training
        querys = []
        predicts = []
        goldens = []
        src_oovs = []
        for batch_data in test_manager.iter_batch():
            query_id,golden_id,predict_id=dev_model.eval_step(sess,batch_data)
            querys.extend(query_id)
            goldens.extend(golden_id)
            predicts.extend(predict_id)

        text = generate_text(predicts, batch_size, Y_MAX_LENGTH, Y_VOCAB_SIZE, y_idx_to_word)
        #print('[INFO]: Batch {} optimized output for Epoch {}:'.format(batch, epoch))
        correct_text= generate_text(goldens, batch_size, Y_MAX_LENGTH, Y_VOCAB_SIZE, y_idx_to_word)
        correct_questions=generate_text(querys, batch_size, X_MAX_LENGTH, X_VOCAB_SIZE, x_idx_to_word)
        correct_num=0
        for _ in range(len(predicts)):
            if _%50==0:
                print(str(correct_text[_])+"#####"+str(text[_])+"#####"+correct_questions[_])
            if correct_text[_]==text[_]:
                correct_num=correct_num+1
        all_num=len(predicts)
        print(str(correct_num)+" "+str(all_num)+" "+str(float(correct_num)/float(all_num)))
        print('\n[STATUS]Epoch {}  complete! '.format(epoch))
        dev_line=str(correct_num)+" "+str(all_num)+" "+str(float(correct_num)/float(all_num))+"\n[STATUS]Epoch "+str(epoch)+"  complete! "
        
        list_test_number=[]
        f= codecs.open(test_number_file, "r", encoding="UTF-8").readlines()
        for line in f:
            list_test_number.append(line.strip())
        file_all= open("all_answers", "w")
        for _ in range(len(text)):
            file_all.write(correct_questions[_]+"\n")
            file_all.write(str(correct_text[_])+"#####"+str(text[_])+"#####"+test_correct_answer[_]+"#####"+list_test_number[_]+"\n")
        gen_ans,corr_ans,ground_ans,check_ans=get_answer(text,correct_text,test_number_file,test_correct_answer)
        correct_num=0
        all_correct=0
        ground_num=0
        all_ground=0
        file_wrong= open("wrong_answers", "w")
        for _ in range(len(gen_ans)):
            #if _%50==0:
            #print(str(gen_ans[_])+"#####"+str(corr_ans[_])+"#####"+str(ground_ans[_]))
            if float(gen_ans[_])==float(corr_ans[_]):
                correct_num=correct_num+1
            else:
                file_wrong.write(str(gen_ans[_])+"#####"+str(ground_ans[_])+"#####"+str(text[_])+"#####"+str(correct_text[_])+"#####"+list_test_number[_].split("###")[6]+"#####"+list_test_number[_].split("###")[1]+"\n")

            if float(corr_ans[_])<=float(ground_ans[_])*1.01 and float(corr_ans[_])>=float(ground_ans[_])*0.99 :
                ground_num=ground_num+1
            else:
                print(correct_questions[_])
                print(str(correct_text[_])+"#####"+str(text[_])+"#####"+test_correct_answer[_])
                print(str(gen_ans[_])+"#####"+str(corr_ans[_])+"#####"+str(ground_ans[_]))

#            else:
#                print(str(gen_ans[_])+"#####"+str(corr_ans[_])+"#####"+str(ground_ans[_]))
#                print(str(correct_text[_])+"#####"+str(text[_])+"#####"+correct_questions[_]+"###"+test_n1_number[_])
        all_num=len(gen_ans)
        print("dev_result"+dev_line)
        print("correct_num"+str(correct_num)+" "+str(all_num)+" "+str(float(correct_num)/float(ground_num)))
        print("actual_correct_num"+str(correct_num)+" "+str(all_num)+" "+str(float(correct_num)/float(all_num)))
        print("ground_num"+str(ground_num)+" "+str(all_num)+" "+str(float(ground_num)/float(all_num)))


if __name__ == '__main__':
    if not FLAGS.test:
        lstm_train()
    else:
        lstm_test()
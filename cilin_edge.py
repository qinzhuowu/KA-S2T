#coding:utf-8
import numpy as np
import sys

import codecs
import collections
import random

from os.path import isfile, join
reload(sys)
sys.setdefaultencoding( "utf-8" )


#如果想让编码题目和解码答案用相同的词表，词表存在Data//mix_vocab里
train_encode_vec = '..//..//GCN_Data//jieba//train_recut.enc'  
train_decode_vec = '..//..//GCN_Data//train_tree.dec'  
valid_encode_vec = '..//..//GCN_Data//jieba//valid_recut.enc'  
valid_decode_vec = '..//..//GCN_Data//valid_tree.dec'
test_encode_vec = '..//..//GCN_Data//jieba//test_recut.enc'  
test_decode_vec = '..//..//GCN_Data//test_tree.dec'
encode_vocab_file="..//..//GCN_Data//jieba//encode_vocabulary_recut"
decode_vocab_file="..//..//GCN_Data//jieba//decode_vocabulary_recut"

#train_number_file='..//..//GCN_Data//train_n1_number'  
#valid_number_file='..//..//GCN_Data//valid_n1_number'  
#test_number_file='..//..//GCN_Data//test_n1_number'

entity_vocab_file="..//..//GCN_Data//jieba//entity_vocabulary"
entity_vocab_size=2622

cilin_file="..//..//GCN_Data//KnowledgeBase//cilin.txt"

cilin_vocab_file="..//..//GCN_Data//KnowledgeBase//cilin_all_vocabulary"
cilin_has_pair_vocab_file="..//..//GCN_Data//KnowledgeBase//cilin_pair_vocabulary"

#a1_list=['A0', 'A1', 'A2', 'A3', 'A4', 'B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'C0', 'C1', 'C2', 'C3', 'C4', 'D0', 'D1', 'D2', 'D3', 'D4', 'E0', 'E1', 'E2', 'E3', 'F0', 'F1', 'F2', 'G0', 'G1', 'H0', 'H1', 'H2', 'I0', 'I1', 'I2', 'I3']
X_VOCAB_SIZE = 4000 
Y_VOCAB_SIZE=26

PAD_ID=0

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

def set_edge(cilin_edge,word_list):
	#print(word_list)
	for word1 in word_list:
		for word2 in word_list:
			if word1!=word2:
				cilin_edge[x_word_to_idx[word1]][x_word_to_idx[word2]]=1
				cilin_edge[x_word_to_idx[word2]][x_word_to_idx[word1]]=1
	return cilin_edge
def get_cilin_edge():
	cilin_edge=np.zeros((X_VOCAB_SIZE, X_VOCAB_SIZE), dtype = np.int32)
	lines = codecs.open(cilin_file, "r", encoding="UTF-8").readlines()
	label_list=[]
	label_word={}
	all_word_list=[]
	need_index_list2=["Ba","Bb","Be","Bg","Bh","Bi","Bl","Bm","Bn","Bo","Bp","Bq","Br","Dk","Dm"]
	need_index_list4=["Cb25","Cb28","Dc01","Dn10"]
	for line in lines:
		line=line.replace('\n','')
		items=line.split()
		index=items[0]
		word_list=[]
		index_2=index[0:2]
		index_4=index[0:4]
		if index_2 in need_index_list2 or index_4 in need_index_list4:
			for word in items[1:]:
				if word in x_idx_to_word:
					word_list.append(word)
					if word not in all_word_list:
						all_word_list.append(word)
			if len(word_list)!=0:
				label_list.append(index)
				label_word[index]=word_list
	with open("label_word.txt", 'w') as f:
		for index in label_list:
			f.write(index+" "+" ".join(label_word[index])+"\n")
	#Aa01A01=
	curr_label=label_list[0]
	curr_group=label_list[0][0:5]
	curr_class=label_list[0][0:4]
	curr_cate=label_list[0][0:2]

	#黄牛 野牛 肉牛
	same_cate_list=[]
	#牛 奶牛 小牛 牦牛 黄牛 野牛 肉牛
	same_group_list=[]
	#牛 羊
	same_class_list=[]
	#熊 牛 羊
	same_category_list=[label_word[curr_label][0]]
	for index in label_list:
		temp_label=index
		temp_group=index[0:5]
		temp_class=index[0:4]
		temp_cate=index[0:2]
		#if temp_label!=curr_label:
		#	cilin_edge=set_edge(cilin_edge,label_word[index])
		#	curr_label=temp_label
		if temp_group!=curr_group:
			cilin_edge=set_edge(cilin_edge,same_group_list)
			curr_group=temp_group
			same_group_list=label_word[index]

			same_category_list.append(label_word[index][0])
		else:
			same_group_list.extend(label_word[index])

		if temp_cate!=curr_cate:
			cilin_edge=set_edge(cilin_edge,same_category_list)
			curr_cate=temp_cate
			same_category_list=[label_word[index][0]]
	cilin_edge=set_edge(cilin_edge,same_group_list)
	cilin_edge=set_edge(cilin_edge,same_category_list)


	with open("cilin_edge.txt", 'w') as f:
		for index in cilin_edge:
			index_new = [str(x) for x in index]
			f.write(" ".join(index_new)+"\n")
	with open("all_word_list.txt", 'w') as f:
		for index in all_word_list:
			f.write(index+"\n")
	return cilin_edge,all_word_list

cilin_edge,all_word_list=get_cilin_edge()


def read_cilin_edge():
	cilin_edge=np.zeros((X_VOCAB_SIZE, X_VOCAB_SIZE), dtype = np.int32)
	edge_data=codecs.open("cilin_edge.txt", "r", encoding="UTF-8").readlines()
	for index1 in range(len(edge_data)):
		line_list=edge_data[index1].strip().split()
		for index2 in range(len(line_list)):
			cilin_edge[index1][index2]=float(line_list[index2])
	all_word_list=[]
	word_data=codecs.open("all_word_list.txt", "r", encoding="UTF-8").readlines()
	for word in word_data:
		all_word_list.append(word.strip())

	return cilin_edge,all_word_list
#cilin_edge,all_word_list=read_cilin_edge()

def check_edge(sen_index_list):
	index_list=[]
	flag=0
	for index1 in sen_index_list:
		flag=0
		for index2 in sen_index_list:
			if cilin_edge[index1][index2]==1:
				if index1 not in index_list:
					index_list.append(index1)
				if index2 not in index_list:
					index_list.append(index2)
				break
		if index1 not in index_list:
			#temp_list=[]
			for index in range(len(cilin_edge[index1])):
				if cilin_edge[index1][index]==1:
					#temp_list.append(index)
					if cilin_edge[index][index2]==1:
						if index1 not in index_list:
							index_list.append(index1)
						if index2 not in index_list:
							index_list.append(index2)
						if index not in index_list:
							index_list.append(index)
						flag=1
						break
		'''
		if flag==0:
			for index in range(len(temp_list)):
				for index_ in range(len(cilin_edge[index])):
					if cilin_edge[index][index_]==1:
						if index1 not in index_list:
							index_list.append(index1)
						if index2 not in index_list:
							index_list.append(index2)
						if index not in index_list:
							index_list.append(index)
						if index_ not in index_list:
							index_list.append(index_)
						flag=1
						break
		'''
	return index_list

def check_line_edge(cilin_edge,word1,word2):
	index1=x_word_to_idx[word1]
	index2=x_word_to_idx[word2]
	flag=0
	if cilin_edge[index1][index2]==1:
		print(word1+" "+word2+" 1")
	else:
		temp_list=[]
		for index in range(len(cilin_edge[index1])):
			if cilin_edge[index1][index]==1:
				if cilin_edge[index][index2]==1:
					print(word1+" "+word2+" "+x_idx_to_word[index]+" 2")
					flag=1
					break
				temp_list.append(index)
		if flag==0:
			for index in range(len(temp_list)):
				for index_ in range(len(cilin_edge[index])):
					if cilin_edge[index][index_]==1:
						if cilin_edge[index_][index2]==1:
							print(word1+" "+word2+" "+x_idx_to_word[index]+" "+x_idx_to_word[index_]+" 3")
							flag=1
		if flag==0:
			print(word1+" "+word2+" no edge")



'''
check_line_edge(cilin_edge,u"牛",u"羊")
check_line_edge(cilin_edge,u"牛",u"猴子")
check_line_edge(cilin_edge,u"老鼠",u"猴")
check_line_edge(cilin_edge,u"老鼠",u"牛")
check_line_edge(cilin_edge,u"牛",u"牛")
check_line_edge(cilin_edge,u"菊花",u"牛")
check_line_edge(cilin_edge,u"菊花",u"月季花")
check_line_edge(cilin_edge,u"菊花",u"花")
'''
def get_node_edge(src_list):
	sen_word_list=[]
	sen_index_list=[]
	for word in src_list:
		if word in all_word_list and word not in sen_word_list:
			sen_word_list.append(word)
			sen_index_list.append(x_word_to_idx[word])
	new_index_list=check_edge(sen_index_list)

	#print(sen_index_list)
	#print(" ".join(x_idx_to_word[x] for x in sen_index_list))

	#print(new_index_list)
	#print(" ".join(x_idx_to_word[x] for x in new_index_list))
	sen_edge_list=[]
	for i in new_index_list:
		temp_list=[]
		for j in new_index_list:
			temp_list.append(cilin_edge[i][j])
		sen_edge_list.append(temp_list)
	return new_index_list,sen_edge_list

#sen_index_list,sen_edge_list=get_node_edge(u"小华 从 家 走到 学校 ， 每分钟 走 n1 米 ， 用 了 n2 分钟 ． 返回 时用 了 n3 分钟 ， 每分钟 走 多少 米 ？".split())
#print(sen_index_list)
#print(sen_edge_list)

def pad_node_edge(src_node_dataset,src_edge_dataset):
	pad_node_batch = []
	pad_edge_batch=[]
	max_src_length = max([len(src) for src in src_node_dataset])
	#print(max_src_length)
	for i in range(len(src_node_dataset)):
		src_id  = src_node_dataset[i]
		paddings = [PAD_ID]*(max_src_length - len(src_id))
		pad_node_batch.append(src_id+paddings)

	for i in range(len(src_edge_dataset)):
		edge_curr=src_edge_dataset[i]
		edge_pad_temp=[]
		need_to_pad=max_src_length - len(edge_curr)
		for j in range(len(edge_curr)):
			edge_id=edge_curr[j]
			need_2_level=max_src_length-len(edge_id)
			#if need_2_level!=need_to_pad:
			#	print(src_node_dataset[i])
			#	print(src_edge_dataset[i])

			paddings=[PAD_ID]*(need_2_level)
			edge_pad_temp.append(edge_id+paddings)

		for j in range(need_to_pad):
			paddings=[PAD_ID]*(max_src_length)
			edge_pad_temp.append(paddings)
		pad_edge_batch.append(edge_pad_temp)
	#print(max_src_length)
	'''
	for i in range(len(src_node_dataset)):
		if len(pad_node_batch[i])!=max_src_length:
			print("wrong src_node_dataset")
	for i in range(len(src_node_dataset)):
		if len(pad_edge_batch[i])!=max_src_length:
			print("wrong src_edge_dataset")
		for j in pad_edge_batch[i]:
			if len(j) !=max_src_length:
				print(len(j))
				print(src_node_dataset[i])
				print(src_edge_dataset[i])
				print("wrong src_edge_dataset 2-level")
	'''
	return pad_node_batch,pad_edge_batch


def get_node_edge_list():
	sen_index_list_dict={}
	sen_edge_list_dict={}
	
	encode_train_dataset = codecs.open(train_encode_vec, "r", encoding="UTF-8").readlines()
	i=0
	for line in encode_train_dataset:
		if i%100==0:
			print(i)
		sen_index_list,sen_edge_list=get_node_edge(line.strip().split(" "))
		sen_index_list_dict[line]=sen_index_list
		sen_edge_list_dict[line]=sen_edge_list
		i=i+1
	encode_valid_dataset = codecs.open(valid_encode_vec, "r", encoding="UTF-8").readlines()
	i=0
	for line in encode_valid_dataset:
		if i%100==0:
			print(i)
		sen_index_list,sen_edge_list=get_node_edge(line.strip().split(" "))
		sen_index_list_dict[line]=sen_index_list
		sen_edge_list_dict[line]=sen_edge_list
		i=i+1
	
	encode_test_dataset = codecs.open(test_encode_vec, "r", encoding="UTF-8").readlines()
	i=0
	for line in encode_test_dataset:
		if i%100==0:
			print(i)
		sen_index_list,sen_edge_list=get_node_edge(line.strip().split(" "))
		sen_index_list_dict[line]=sen_index_list
		sen_edge_list_dict[line]=sen_edge_list
		i=i+1
	with open("sen_index_list_dict.txt", 'w') as f:
		for index in sen_index_list_dict:
			index_new=[str(x) for x in sen_index_list_dict[index]]
			f.write(index.strip()+"###"+" ".join(index_new)+"\n")
	with open("sen_word_list_dict.txt", 'w') as f:
		for index in sen_index_list_dict:
			index_new=[str(x_idx_to_word[x]) for x in sen_index_list_dict[index]]
			f.write(index.strip()+"###"+" ".join(index_new)+"\n")
	with open("sen_edge_list_dict.txt", 'w') as f:
		for index in sen_edge_list_dict:
			f.write(index.strip()+"###")
			edge_new=[]
			for edge_list in sen_edge_list_dict[index]:
				edge_list_new = [str(x) for x in edge_list]
				edge_new.append(" ".join(edge_list_new))
			f.write("***".join(edge_new)+"\n")
	return sen_index_list_dict,sen_edge_list_dict
#sen_index_list_dict,sen_edge_list_dict=get_node_edge_list()

def get_node_edge_from_file():
	sen_index_list_dict={}
	sen_edge_list_dict={}
	
	encode_train_dataset = codecs.open("sen_index_list_dict.txt", "r", encoding="UTF-8").readlines()
	for line in encode_train_dataset:
		list_temp=[]
		index_list=line.strip().split("###")
		for index in index_list[1].split():
			list_temp.append(int(index))
		sen_index_list_dict[index_list[0]]=list_temp
	encode_train_dataset = codecs.open("sen_edge_list_dict.txt", "r", encoding="UTF-8").readlines()
	for line in encode_train_dataset:
		list_temp=[]
		index_list=line.strip().split("###")
		for index_line in index_list[1].split("***"):
			edge_temp=[]
			for index in index_line.split():
				edge_temp.append(int(index))
			list_temp.append(edge_temp)
		sen_edge_list_dict[index_list[0]]=list_temp
	num=0
	for line in sen_index_list_dict:
		index_list=sen_index_list_dict[line]
		edge_list=sen_edge_list_dict[line]
		if len(index_list)<1:
			#print(line)
			num+=1
	print(num)
	
	return sen_index_list_dict,sen_edge_list_dict

sen_index_list_dict,sen_edge_list_dict=get_node_edge_from_file()
def get_node_edge_from_dict(src):
	return sen_index_list_dict[src],sen_edge_list_dict[src]
#sen_index_list,sen_edge_list=get_node_edge_from_dict(src)
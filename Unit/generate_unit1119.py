#coding:utf-8
import numpy as np

import codecs
import random

import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )


#如果想让编码题目和解码答案用相同的词表，词表存在Data//mix_vocab里
train_encode_vec = 'train.enc'  
train_decode_vec = 'train_tree.dec'  
valid_encode_vec = 'valid.enc'  
valid_decode_vec = 'valid_tree.dec'
test_encode_vec = 'test.enc'  
test_decode_vec = 'test_tree.dec'
encode_vocab_file="encode_vocabulary_tree"
decode_vocab_file="decode_vocabulary_tree"



train_number_file='train_n1_number'  
valid_number_file='valid_n1_number'  
test_number_file='test_n1_number'  

train_unit_file='Unit//train_number_unit'
valid_unit_file='Unit//valid_number_unit'
test_unit_file='Unit//test_number_unit'


train_check='train_unit'
valid_check='valid_unit'
test_check='test_unit'


unit_vocab_file="Unit//unit_vocabulary+-"
unit_vocab_file1="Unit//unit_vocabulary×"
not_unit_vocab_file="Unit//not_unit_vocabulary"
def get_unit_vocab(filename):
    x_idx_to_word=[]
    x_word_to_idx={}
    encode_vocab_dataset = codecs.open(filename, "r", encoding="UTF-8").readlines()
    for line in encode_vocab_dataset:
        if "###" in line:
            list_word = line.strip().split("###")
            for word in list_word:
                x_idx_to_word.append(word)
                x_word_to_idx[word]=list_word[0]
        else:
            x_idx_to_word.append(line.strip())
            x_word_to_idx[line.strip()]=line.strip()

    x_idx_to_word.sort(key = lambda i:len(i),reverse=True) 
    return x_idx_to_word, x_word_to_idx

unit_word_list, unit_syno_to_word=get_unit_vocab(unit_vocab_file)
not_unit_word_list, not_unit_syno_to_word=get_unit_vocab(not_unit_vocab_file)

def multi_devide(question_line,n_number):
    str_multi="*/:"
    number_list=[]
    list_front=["的","占","了"]
    list_after=[",",".","，","。"]
    question_line_list=question_line.split()
    for i in range(1,len(question_line_list)-1):
        if question_line_list[i] in n_number:
            if question_line_list[i+1]=="倍":
                number_list.append(question_line_list[i])
            elif question_line_list[i-1] in list_front  and question_line_list[i+1] not in unit_word_list:
                number_list.append(question_line_list[i])
    '''
    if "每" in question_line_list:
        index1=question_line_list.index("每")
        for i in range(4):
            if index1+2+i<len(question_line_list) and question_line_list[index1+1+i] in n_number:
                number_list.append(question_line_list[index1+i+1])
                break
    '''
    str_multi+=" ".join(number_list)
    return str_multi

unit_vocab_file1="Unit//unit_vocabulary×"
f1=open("Unit//not_get_unit","w")
def get_number_unit(number_file,encode_file,output_file):
    number_dataset = codecs.open(number_file, "r", encoding="UTF-8").readlines()
    encode_dataset = codecs.open(encode_file, "r", encoding="UTF-8").readlines()
    output=open(output_file,"w")
    question_list=[]
    answer_list=[]
    for line in encode_dataset:
        question_list.append(line.strip())
    i=0
    number_list=[]

    count_number=0
    count_solve_number=0
    count_sentence=0
    for i in range(len(number_dataset)):
        number_n1_list=number_dataset[i].split("###")
        answer_list.append(number_n1_list[5].strip())
        number_list.append(number_n1_list[6].strip())

    for i in range(len(question_list)):
        not_solve_flag=""
        n_number={}
        str_number_unit_list=[]
        line_number_list=number_list[i].split("*")[:-1]
        for word in line_number_list:
            temp_list=word.split(" ")
            n1=temp_list[0]
            number1=temp_list[1]
            n_number[n1]=number1

        #
        str_multi=multi_devide(question_list[i],n_number)
        for n1 in n_number:
            number1=n_number[n1]
            index1=question_list[i].find(n1)
            if index1==-1:
                print("find not exist n1")
                print(question_list[i])
                print(number_list[i])
                str_number_unit_list.append(n1+"***"+number1+"***"+"__UNK__")
            else:
                str_temp=question_list[i][index1+len(n1)+1:]
                unit_temp="__UNK__"
                for word in unit_word_list:
                    word1=word+" "
                    if str_temp.startswith(word1):
                        unit_temp=unit_syno_to_word[word]
                        break
                if unit_temp!="__UNK__":
                    str_number_unit_list.append(n1+"***"+number1+"***"+unit_temp)
                    count_solve_number+=1
                else:
                    #not_unit 的 倍
                    for word in not_unit_word_list:
                        word1=word+" "
                        if str_temp.startswith(word1) or str_temp==word:
                            unit_temp=not_unit_syno_to_word[word]
                            break
                    if unit_temp=="__UNK__" and str_temp!="":
                        not_solve_flag+=n1+" "
                    str_number_unit_list.append(n1+"***"+number1+"***"+"__UNK__")
            count_number+=1
        str_number_unit_line="###".join(str_number_unit_list)
        output.write(question_list[i]+"\n")
        output.write(answer_list[i]+"\n")
        output.write(str_number_unit_line+"\n")
        output.write(str_multi+"\n")
        if not_solve_flag !="":            
            f1.write(question_list[i]+"\n")
            f1.write(number_list[i]+"###"+not_solve_flag+"\n")
            count_sentence+=1
    print(str(count_solve_number)+" "+str(count_number)+" "+str(count_sentence))

get_number_unit(train_number_file,train_encode_vec,train_unit_file)
get_number_unit(valid_number_file,valid_encode_vec,valid_unit_file)
get_number_unit(test_number_file,test_encode_vec,test_unit_file)

def get_unit_vocab(filename):
    x_idx_to_word=[]
    x_word_to_idx={}
    encode_vocab_dataset = codecs.open(filename, "r", encoding="UTF-8").readlines()
    for line in encode_vocab_dataset:
        if "###" in line:
            list_word = line.strip().split("###")
            for word in list_word:
                x_idx_to_word.append(word)
                x_word_to_idx[word]=list_word[0]
        else:
            x_idx_to_word.append(line.strip())
            x_word_to_idx[line.strip()]=line.strip()

    x_idx_to_word.sort(key = lambda i:len(i),reverse=True) 
    return x_idx_to_word, x_word_to_idx

unit_word_list, unit_syno_to_word=get_unit_vocab(unit_vocab_file)
not_unit_word_list, not_unit_syno_to_word=get_unit_vocab(not_unit_vocab_file)


operator_unit = {
    '+' : 1,
    '-' : 2,
    '*' : 3,
    '/' : 4,
    '^' : 5
}

def ans2ids(sentences,nm_word,unit_constraint):
    '''
    ids = []
    for w in sentences:
        _id = y_word_to_idx.get(w,UNK_ID)
        ids.append(_id)
    '''
    tree_list=sentences.split()

    i=0
    pad_str={}
    multi_right=""
    multi_wrong=""
    count_right=0
    count_all=0
    for i in range(2,len(tree_list)):
        if tree_list[i] in operator_unit and tree_list[i-2] not in operator_unit and tree_list[i-1] not in operator_unit:
            str_word = tree_list[i-2]+" "+tree_list[i-1]
            if str_word in unit_constraint:
                if unit_constraint[str_word]=="*":
                    if tree_list[i] =="+" or tree_list[i] =="-":
                        multi_wrong+=str_word+" "+tree_list[i]+"###"
                    else:
                        count_right+=1
                elif unit_constraint[str_word]=="+":
                    if tree_list[i] =="+" or tree_list[i] =="-":
                        count_right+=1
                    else:
                        multi_wrong+=str_word+" "+tree_list[i]+"###"
                count_all+=1

    return pad_str,multi_wrong,count_right,count_all




#f1=open("not_pass_check","w")

#f2=open("not_pass_*","w")
f3=open("not_pass_+-*","w")
#如果n1和n2之间是+-单位，且n1和n2都有单位，那他们的单位一定相同
#如果n1是几分之一，n2不是，那必须用×/来处理它们
#如果n1,n2都是几分之一，那必须用+-来处理它们

def check_unit_const(input_file,output_file):
    number_dataset = codecs.open(input_file, "r", encoding="UTF-8").readlines()
    output=open(output_file,"w")
    question_list=[]
    answer_list=[]
    number_line_list=[]
    multi_line_list=[]
    i=0
    for line in number_dataset:
        if i%4==0:
            question_list.append(line.strip())
        elif i%4==1:
            answer_list.append(line.strip())
        elif i%4==2:
            number_line_list.append(line.strip())
        else:
            multi_line_list.append(line.strip())
        i+=1
    
    count_avg_len=0
    count_max_len=0
    count_all=0
    multi_right=0
    multi_all=0

    count_use=0
    count_all_que=0
    for i in range(len(question_list)):
        number_line=number_line_list[i].split("###")
        nm_word=[]
        nm_unit=[]
        for word in number_line:
            nm_word.append(word.split("***")[0])
            nm_unit.append(word.split("***")[2])
        multi_word_line=multi_line_list[i].split(":")[1]
        multi_word_list=[]
        if multi_word_line !="":
            multi_word_list=multi_word_line.split()

        unit_constraint={}
        for index1 in range(len(nm_word)):
            for index2 in range(len(nm_word)):
                if index1!=index2:
                    word1=nm_word[index1]
                    word2=nm_word[index2]
                    if word2 in multi_word_list :
                        if word1 not in multi_word_list:
                            unit_constraint[word1+" "+word2]="*"
                        else:
                            unit_constraint[word1+" "+word2]="+"
                        unit_constraint["1 "+word2]="+"
                    elif nm_unit[index1]!="__UNK__" and nm_unit[index2]!="__UNK__" and nm_unit[index1]!= nm_unit[index2]:
                        unit_constraint[word1+" "+word2]="*"
                        unit_constraint[word2+" "+word1]="*"
                    elif nm_unit[index1]!="__UNK__" and nm_unit[index2]!="__UNK__" and nm_unit[index1]== nm_unit[index2]:
                        unit_constraint[word1+" "+word2]="+"
                        unit_constraint[word2+" "+word1]="+"
        pad_str,multi_wrong,line_right,line_all=ans2ids(answer_list[i],nm_word,unit_constraint)
        multi_right+=line_right
        multi_all+=line_all
        output.write(question_list[i]+"\n")
        output.write(answer_list[i]+"\n")
        constraint_line=""
        if len(unit_constraint)!=0:
            constraint_line="###".join(str_pad+" "+str(unit_constraint[str_pad]) for str_pad in unit_constraint)
            count_use+=1
        else:
            constraint_line="__UNK__"
        output.write(constraint_line+"\n")
        if len(unit_constraint)>count_max_len:
            count_max_len=len(unit_constraint)

        if len(unit_constraint)>40:
            print(constraint_line)
        count_avg_len+=len(unit_constraint)

        if multi_wrong!="":
            f3.write(question_list[i]+"\n")
            f3.write(answer_list[i]+"\n")
            f3.write(multi_wrong+"\n")

        count_all_que+=1
    #print(str(count_check)+"  "+str(count_all)+" "+str(float(count_check)/float(count_all)))
    #print(str(multi_right)+"  "+str(multi_all)+" "+str(float(multi_right)/float(multi_all)))
    print(str(count_use)+"  "+str(count_all_que)+" "+str(float(count_use)/float(count_all_que)))
    print(str(count_max_len)+"  "+str(count_avg_len)+" "+str(count_use)+" "+str(float(count_avg_len)/float(count_use)))

'''
#这个是认为所有同样单位都只能是+-，两个分米求面积
def check_unit_const(input_file,output_file):
    number_dataset = codecs.open(input_file, "r", encoding="UTF-8").readlines()
    output=open(output_file,"w")
    question_list=[]
    answer_list=[]
    number_line_list=[]
    i=0
    for line in number_dataset:
        if i%3==0:
            question_list.append(line.strip())
        elif i%3==1:
            answer_list.append(line.strip())
        else:
            number_line_list.append(line.strip())
        i+=1
    
    count_check=0
    count_all=0

    for i in range(len(question_list)):
        number_line=number_line_list[i].split("###")
        nm_word=[]
        nm_unit=[]
        for word in number_line:
            nm_word.append(word.split("***")[0])
            nm_unit.append(word.split("***")[2])
        pad_str=ans2ids(answer_list[i],nm_word)
        flag_right=0
        flag_wrong=""
        for id_1 in range(len(nm_word)):
            if nm_unit[id_1]!="__UNK__":
                for id_2 in range(id_1+1,len(nm_word)):
                    if nm_unit[id_1] ==nm_unit[id_2]:
                        if pad_str[id_1][id_2]==1 or pad_str[id_1][id_2]==2:
                            flag_right=1
                            count_check+=1
                        elif pad_str[id_1][id_2]!=0:
                            flag_wrong+=nm_word[id_1]+"  "+nm_word[id_2]+"   "+nm_unit[id_1]+"  "+nm_unit[id_2]+"  "+str(pad_str[id_1][id_2])+"###"
                        if pad_str[id_1][id_2]!=0:
                            count_all+=1
        if flag_wrong!="":
            f1.write(question_list[i]+"\n")
            f1.write(answer_list[i]+"\n")
            f1.write(number_line_list[i]+"\n"+flag_wrong+"\n")
            flag_wrong=""
            for id_i in range(len(nm_word)):
                for id_j in range(id_i+1,len(nm_word)):
                    if pad_str[id_i][id_j]!=0:
                        flag_wrong+=nm_word[id_i]+"  "+nm_word[id_j]+"   "+str(pad_str[id_i][id_j])+"***"
            f1.write(" ".join(nm_word)+"***"+" ".join(nm_unit)+"***"+flag_wrong+"\n")
                                        
        elif flag_right==1:
            output.write(question_list[i]+"\n")
            output.write(answer_list[i]+"\n")
            output.write(number_line_list[i]+"\n")
            flag_wrong=""
            for id_i in range(len(nm_word)):
                for id_j in range(id_i+1,len(nm_word)):
                    if pad_str[id_i][id_j]!=0:
                        flag_wrong+=nm_word[id_i]+"  "+nm_word[id_j]+"   "+str(pad_str[id_i][id_j])+"***"
            output.write(" ".join(nm_word)+"***"+" ".join(nm_unit)+"***"+flag_wrong+"\n")
    print(str(count_check)+"  "+str(count_all))
'''

check_unit_const(train_unit_file,train_check)
check_unit_const(valid_unit_file,valid_check)
check_unit_const(test_unit_file,test_check)

'''
#这个是认为所有同样单位都只能是+-，两个分米求面积
def check_unit_const(input_file,output_file):
    number_dataset = codecs.open(input_file, "r", encoding="UTF-8").readlines()
    output=open(output_file,"w")
    question_list=[]
    answer_list=[]
    number_line_list=[]
    i=0
    for line in number_dataset:
        if i%3==0:
            question_list.append(line.strip())
        elif i%3==1:
            answer_list.append(line.strip())
        else:
            number_line_list.append(line.strip())
        i+=1
    
    count_check=0
    count_all=0

    for i in range(len(question_list)):
        number_line=number_line_list[i].split("###")
        nm_word=[]
        nm_unit=[]
        for word in number_line:
            nm_word.append(word.split("***")[0])
            nm_unit.append(word.split("***")[2])
        pad_str=ans2ids(answer_list[i],nm_word)
        flag_right=0
        flag_wrong=""
        for id_1 in range(len(nm_word)):
            if nm_unit[id_1]!="__UNK__":
                for id_2 in range(id_1+1,len(nm_word)):
                    if nm_unit[id_1] ==nm_unit[id_2]:
                        if pad_str[id_1][id_2]==1 or pad_str[id_1][id_2]==2:
                            flag_right=1
                            count_check+=1
                        elif pad_str[id_1][id_2]!=0:
                            flag_wrong+=nm_word[id_1]+"  "+nm_word[id_2]+"   "+nm_unit[id_1]+"  "+nm_unit[id_2]+"  "+str(pad_str[id_1][id_2])+"###"
                        if pad_str[id_1][id_2]!=0:
                            count_all+=1
        if flag_wrong!="":
            f1.write(question_list[i]+"\n")
            f1.write(answer_list[i]+"\n")
            f1.write(number_line_list[i]+"\n"+flag_wrong+"\n")
            flag_wrong=""
            for id_i in range(len(nm_word)):
                for id_j in range(id_i+1,len(nm_word)):
                    if pad_str[id_i][id_j]!=0:
                        flag_wrong+=nm_word[id_i]+"  "+nm_word[id_j]+"   "+str(pad_str[id_i][id_j])+"***"
            f1.write(" ".join(nm_word)+"***"+" ".join(nm_unit)+"***"+flag_wrong+"\n")
                                        
        elif flag_right==1:
            output.write(question_list[i]+"\n")
            output.write(answer_list[i]+"\n")
            output.write(number_line_list[i]+"\n")
            flag_wrong=""
            for id_i in range(len(nm_word)):
                for id_j in range(id_i+1,len(nm_word)):
                    if pad_str[id_i][id_j]!=0:
                        flag_wrong+=nm_word[id_i]+"  "+nm_word[id_j]+"   "+str(pad_str[id_i][id_j])+"***"
            output.write(" ".join(nm_word)+"***"+" ".join(nm_unit)+"***"+flag_wrong+"\n")
    print(str(count_check)+"  "+str(count_all))
'''
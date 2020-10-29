# coding: utf-8
import random
import json
import copy
import re
import time
import math
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

PAD_token = 0

cons_mode=[1,2,3]

class Lang:
    """
    class to save the vocab and two dict: the word->index and index->word
    """
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  # Count word tokens
        self.num_start = 0

    def add_sen_to_vocab(self, sentence):  # add words of sentence to vocab
        for word in sentence:
            if re.search("N\d+|NUM|\d+", word):
                continue
            if word not in self.index2word:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word.append(word)
                self.n_words += 1
            else:
                self.word2count[word] += 1

    def trim(self, min_count):  # trim words below a certain count threshold
        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.index2word), len(keep_words) / len(self.index2word)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  # Count default tokens

        for word in keep_words:
            self.word2index[word] = self.n_words
            self.index2word.append(word)
            self.n_words += 1

    def build_input_lang(self, trim_min_count):  # build the input lang vocab and dict
        if trim_min_count > 0:
            self.trim(trim_min_count)
            self.index2word = ["PAD", "NUM", "UNK"] + self.index2word
        else:
            self.index2word = ["PAD", "NUM"] + self.index2word
        self.word2index = {}
        self.n_words = len(self.index2word)
        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

    def build_output_lang(self, generate_num, copy_nums):  # build the output lang vocab and dict
        self.index2word = ["PAD", "EOS"] + self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)] +\
                          ["SOS", "UNK"]
        self.n_words = len(self.index2word)
        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

    def build_output_lang_for_tree(self, generate_num, copy_nums):  # build the output lang vocab and dict
        self.num_start = len(self.index2word)

        self.index2word = self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)] + ["UNK"]
        self.n_words = len(self.index2word)

        for i, j in enumerate(self.index2word):
            self.word2index[j] = i


def load_raw_data(filename):  # load the json data to list(dict()) for MATH 23K
    print("Reading lines...")
    f = open(filename,'r')
    js = ""
    data = []
    for i, s in enumerate(f):
        js += s
        i += 1
        if i % 7 == 0:  # every 7 line is a json
            data_d = json.loads(js)
            if "千米/小时" in data_d["equation"]:
                data_d["equation"] = data_d["equation"][:-5]
            data.append(data_d)
            js = ""

    return data


# remove the superfluous brackets
def remove_brackets(x):
    y = x
    if x[0] == "(" and x[-1] == ")":
        x = x[1:-1]
        flag = True
        count = 0
        for s in x:
            if s == ")":
                count -= 1
                if count < 0:
                    flag = False
                    break
            elif s == "(":
                count += 1
        if flag:
            return x
    return y


def load_mawps_data(filename):  # load the json data to list(dict()) for MAWPS
    print("Reading lines...")
    f = open(filename,'r')
    data = json.load(f)
    out_data = []
    for d in data:
        if "lEquations" not in d or len(d["lEquations"]) != 1:
            continue
        x = d["lEquations"][0].replace(" ", "")

        if "lQueryVars" in d and len(d["lQueryVars"]) == 1:
            v = d["lQueryVars"][0]
            if v + "=" == x[:len(v)+1]:
                xt = x[len(v)+1:]
                if len(set(xt) - set("0123456789.+-*/()")) == 0:
                    temp = d.copy()
                    temp["lEquations"] = xt
                    out_data.append(temp)
                    continue

            if "=" + v == x[-len(v)-1:]:
                xt = x[:-len(v)-1]
                if len(set(xt) - set("0123456789.+-*/()")) == 0:
                    temp = d.copy()
                    temp["lEquations"] = xt
                    out_data.append(temp)
                    continue

        if len(set(x) - set("0123456789.+-*/()=xX")) != 0:
            continue

        if x[:2] == "x=" or x[:2] == "X=":
            if len(set(x[2:]) - set("0123456789.+-*/()")) == 0:
                temp = d.copy()
                temp["lEquations"] = x[2:]
                out_data.append(temp)
                continue
        if x[-2:] == "=x" or x[-2:] == "=X":
            if len(set(x[:-2]) - set("0123456789.+-*/()")) == 0:
                temp = d.copy()
                temp["lEquations"] = x[:-2]
                out_data.append(temp)
                continue
    return out_data


def load_roth_data(filename):  # load the json data to dict(dict()) for roth data
    print("Reading lines...")
    f = open(filename,'r')
    data = json.load(f)
    out_data = {}
    for d in data:
        if "lEquations" not in d or len(d["lEquations"]) != 1:
            continue
        x = d["lEquations"][0].replace(" ", "")

        if "lQueryVars" in d and len(d["lQueryVars"]) == 1:
            v = d["lQueryVars"][0]
            if v + "=" == x[:len(v)+1]:
                xt = x[len(v)+1:]
                if len(set(xt) - set("0123456789.+-*/()")) == 0:
                    temp = d.copy()
                    temp["lEquations"] = remove_brackets(xt)
                    y = temp["sQuestion"]
                    seg = y.strip().split(" ")
                    temp_y = ""
                    for s in seg:
                        if len(s) > 1 and (s[-1] == "," or s[-1] == "." or s[-1] == "?"):
                            temp_y += s[:-1] + " " + s[-1:] + " "
                        else:
                            temp_y += s + " "
                    temp["sQuestion"] = temp_y[:-1]
                    out_data[temp["iIndex"]] = temp
                    continue

            if "=" + v == x[-len(v)-1:]:
                xt = x[:-len(v)-1]
                if len(set(xt) - set("0123456789.+-*/()")) == 0:
                    temp = d.copy()
                    temp["lEquations"] = remove_brackets(xt)
                    y = temp["sQuestion"]
                    seg = y.strip().split(" ")
                    temp_y = ""
                    for s in seg:
                        if len(s) > 1 and (s[-1] == "," or s[-1] == "." or s[-1] == "?"):
                            temp_y += s[:-1] + " " + s[-1:] + " "
                        else:
                            temp_y += s + " "
                    temp["sQuestion"] = temp_y[:-1]
                    out_data[temp["iIndex"]] = temp
                    continue

        if len(set(x) - set("0123456789.+-*/()=xX")) != 0:
            continue

        if x[:2] == "x=" or x[:2] == "X=":
            if len(set(x[2:]) - set("0123456789.+-*/()")) == 0:
                temp = d.copy()
                temp["lEquations"] = remove_brackets(x[2:])
                y = temp["sQuestion"]
                seg = y.strip().split(" ")
                temp_y = ""
                for s in seg:
                    if len(s) > 1 and (s[-1] == "," or s[-1] == "." or s[-1] == "?"):
                        temp_y += s[:-1] + " " + s[-1:] + " "
                    else:
                        temp_y += s + " "
                temp["sQuestion"] = temp_y[:-1]
                out_data[temp["iIndex"]] = temp
                continue
        if x[-2:] == "=x" or x[-2:] == "=X":
            if len(set(x[:-2]) - set("0123456789.+-*/()")) == 0:
                temp = d.copy()
                temp["lEquations"] = remove_brackets(x[2:])
                y = temp["sQuestion"]
                seg = y.strip().split(" ")
                temp_y = ""
                for s in seg:
                    if len(s) > 1 and (s[-1] == "," or s[-1] == "." or s[-1] == "?"):
                        temp_y += s[:-1] + " " + s[-1:] + " "
                    else:
                        temp_y += s + " "
                temp["sQuestion"] = temp_y[:-1]
                out_data[temp["iIndex"]] = temp
                continue
    return out_data

# for testing equation
# def out_equation(test, num_list):
#     test_str = ""
#     for c in test:
#         if c[0] == "N":
#             x = num_list[int(c[1:])]
#             if x[-1] == "%":
#                 test_str += "(" + x[:-1] + "/100.0" + ")"
#             else:
#                 test_str += x
#         elif c == "^":
#             test_str += "**"
#         elif c == "[":
#             test_str += "("
#         elif c == "]":
#             test_str += ")"
#         else:
#             test_str += c
#     return test_str


def transfer_num(data):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    count_empty=0

    UNK2word_vocab={}
    input1=open("data//UNK2word_vocab","r").readlines()
    for word in input1:
        UNK2word_vocab[word.strip().split("###")[0]]=word.strip().split("###")[1]

    for d in data:
        nums = []
        input_seq = []
        seg_line = d["segmented_text"].encode("UTF-8").strip()
        for UNK_word in UNK2word_vocab:
            if UNK_word in seg_line:
                seg_line=seg_line.replace(UNK_word,UNK2word_vocab[UNK_word])
        seg=seg_line.split(" ")
        equations = d["equation"][2:]

        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start(): pos.end()])
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                if len(s)>0:
                    input_seq.append(s)
                else:
                    count_empty=count_empty+1
        if copy_nums < len(nums):
            copy_nums = len(nums)

        nums_fraction = []

        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1:
                        res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
        assert len(nums) == len(num_pos)

        #unit_list,rule3_list=get_constraint_unit(input_seq,num_pos)
        #new_input_seq,new_num_pos =get_new_inputseq(input_seq,unit_list,rule3_list,num_pos)
        # pairs.append((input_seq, out_seq, nums, num_pos, d["ans"]))
        #* N0 N1 + N0 N2 * N1 N2 爱心 超市 运 来 NUM 千克 大米 ， 卖 了 NUM 天 后 ， 还 剩 NUM 千克 ， 平均 每天 卖 大米 多少 千克 ？
        #print(" ".join(new_input_seq))
        pairs.append((input_seq, out_seq, nums, num_pos))
    print("count_empty")
    print(count_empty)
    temp_g = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)
    return pairs, temp_g, copy_nums

def get_new_inputseq(input_seq,unit_list,rule3_list,num_pos):
    #rule3_list [1,2] whether this pos is a multiple NUM
    #unit_list  [排###个###排] unit for each num
    cons_word_list=[]
    for i in range(0,len(num_pos)-1):
        if 3 in cons_mode:
            if i in rule3_list:
                cons_word_list.append("/")
                N1_word="N"+str(i)
                N2_word="N"+str(i)
                cons_word_list.append(N1_word)
                cons_word_list.append(N2_word)
        for j in range(i+1,len(num_pos)):
            if i !=j:
                if 1 in cons_mode:
                    if unit_list[i]!="" and unit_list[j]!="" and unit_list[i]==unit_list[j]:
                        cons_word_list.append("+")
                        N1_word="N"+str(i)
                        N2_word="N"+str(j)
                        cons_word_list.append(N1_word)
                        cons_word_list.append(N2_word)
                if 2 in cons_mode:
                    if unit_list[i]!="" and unit_list[j]!="" and unit_list[i]!=unit_list[j]:
                        cons_word_list.append("*")
                        N1_word="N"+str(i)
                        N2_word="N"+str(j)
                        cons_word_list.append(N1_word)
                        cons_word_list.append(N2_word)
    input_seq=cons_word_list+input_seq
    new_num_pos = []
    for i, j in enumerate(input_seq):
        if j == "NUM":
            new_num_pos.append(i)
    assert len(new_num_pos) == len(num_pos) 
    return input_seq,new_num_pos   


def transfer_english_num(data):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d+,\d+|\d+\.\d+|\d+")
    pairs = []
    generate_nums = {}
    copy_nums = 0
    for d in data:
        nums = []
        input_seq = []
        seg = d["sQuestion"].strip().split(" ")
        equations = d["lEquations"]

        for s in seg:
            pos = re.search(pattern, s)
            if pos:
                if pos.start() > 0:
                    input_seq.append(s[:pos.start()])
                num = s[pos.start(): pos.end()]
                # if num[-2:] == ".0":
                #     num = num[:-2]
                # if "." in num and num[-1] == "0":
                #     num = num[:-1]
                nums.append(num.replace(",", ""))
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)

        if copy_nums < len(nums):
            copy_nums = len(nums)
        eq_segs = []
        temp_eq = ""
        for e in equations:
            if e not in "()+-*/":
                temp_eq += e
            elif temp_eq != "":
                count_eq = []
                for n_idx, n in enumerate(nums):
                    if abs(float(n) - float(temp_eq)) < 1e-4:
                        count_eq.append(n_idx)
                        if n != temp_eq:
                            nums[n_idx] = temp_eq
                if len(count_eq) == 0:
                    flag = True
                    for gn in generate_nums:
                        if abs(float(gn) - float(temp_eq)) < 1e-4:
                            generate_nums[gn] += 1
                            if temp_eq != gn:
                                temp_eq = gn
                            flag = False
                    if flag:
                        generate_nums[temp_eq] = 0
                    eq_segs.append(temp_eq)
                elif len(count_eq) == 1:
                    eq_segs.append("N"+str(count_eq[0]))
                else:
                    eq_segs.append(temp_eq)
                eq_segs.append(e)
                temp_eq = ""
            else:
                eq_segs.append(e)
        if temp_eq != "":
            count_eq = []
            for n_idx, n in enumerate(nums):
                if abs(float(n) - float(temp_eq)) < 1e-4:
                    count_eq.append(n_idx)
                    if n != temp_eq:
                        nums[n_idx] = temp_eq
            if len(count_eq) == 0:
                flag = True
                for gn in generate_nums:
                    if abs(float(gn) - float(temp_eq)) < 1e-4:
                        generate_nums[gn] += 1
                        if temp_eq != gn:
                            temp_eq = gn
                        flag = False
                if flag:
                    generate_nums[temp_eq] = 0
                eq_segs.append(temp_eq)
            elif len(count_eq) == 1:
                eq_segs.append("N" + str(count_eq[0]))
            else:
                eq_segs.append(temp_eq)

        # def seg_and_tag(st):  # seg the equation and tag the num
        #     res = []
        #     pos_st = re.search(pattern, st)
        #     if pos_st:
        #         p_start = pos_st.start()
        #         p_end = pos_st.end()
        #         if p_start > 0:
        #             res += seg_and_tag(st[:p_start])
        #         st_num = st[p_start:p_end]
        #         if st_num[-2:] == ".0":
        #             st_num = st_num[:-2]
        #         if "." in st_num and st_num[-1] == "0":
        #             st_num = st_num[:-1]
        #         if nums.count(st_num) == 1:
        #             res.append("N"+str(nums.index(st_num)))
        #         else:
        #             res.append(st_num)
        #         if p_end < len(st):
        #             res += seg_and_tag(st[p_end:])
        #     else:
        #         for sst in st:
        #             res.append(sst)
        #     return res
        # out_seq = seg_and_tag(equations)

        # for s in out_seq:  # tag the num which is generated
        #     if s[0].isdigit() and s not in generate_nums and s not in nums:
        #         generate_nums.append(s)
        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
        if len(nums) != 0:
            pairs.append((input_seq, eq_segs, nums, num_pos))

    temp_g = []
    for g in generate_nums:
        if generate_nums[g] >= 5:
            temp_g.append(g)

    return pairs, temp_g, copy_nums


def transfer_roth_num(data):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d+,\d+|\d+\.\d+|\d+")
    pairs = {}
    generate_nums = {}
    copy_nums = 0
    for key in data:
        d = data[key]
        nums = []
        input_seq = []
        seg = d["sQuestion"].strip().split(" ")
        equations = d["lEquations"]

        for s in seg:
            pos = re.search(pattern, s)
            if pos:
                if pos.start() > 0:
                    input_seq.append(s[:pos.start()])
                num = s[pos.start(): pos.end()]
                # if num[-2:] == ".0":
                #     num = num[:-2]
                # if "." in num and num[-1] == "0":
                #     num = num[:-1]
                nums.append(num.replace(",", ""))
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)

        if copy_nums < len(nums):
            copy_nums = len(nums)
        eq_segs = []
        temp_eq = ""
        for e in equations:
            if e not in "()+-*/":
                temp_eq += e
            elif temp_eq != "":
                count_eq = []
                for n_idx, n in enumerate(nums):
                    if abs(float(n) - float(temp_eq)) < 1e-4:
                        count_eq.append(n_idx)
                        if n != temp_eq:
                            nums[n_idx] = temp_eq
                if len(count_eq) == 0:
                    flag = True
                    for gn in generate_nums:
                        if abs(float(gn) - float(temp_eq)) < 1e-4:
                            generate_nums[gn] += 1
                            if temp_eq != gn:
                                temp_eq = gn
                            flag = False
                    if flag:
                        generate_nums[temp_eq] = 0
                    eq_segs.append(temp_eq)
                elif len(count_eq) == 1:
                    eq_segs.append("N"+str(count_eq[0]))
                else:
                    eq_segs.append(temp_eq)
                eq_segs.append(e)
                temp_eq = ""
            else:
                eq_segs.append(e)
        if temp_eq != "":
            count_eq = []
            for n_idx, n in enumerate(nums):
                if abs(float(n) - float(temp_eq)) < 1e-4:
                    count_eq.append(n_idx)
                    if n != temp_eq:
                        nums[n_idx] = temp_eq
            if len(count_eq) == 0:
                flag = True
                for gn in generate_nums:
                    if abs(float(gn) - float(temp_eq)) < 1e-4:
                        generate_nums[gn] += 1
                        if temp_eq != gn:
                            temp_eq = gn
                        flag = False
                if flag:
                    generate_nums[temp_eq] = 0
                eq_segs.append(temp_eq)
            elif len(count_eq) == 1:
                eq_segs.append("N" + str(count_eq[0]))
            else:
                eq_segs.append(temp_eq)

        # def seg_and_tag(st):  # seg the equation and tag the num
        #     res = []
        #     pos_st = re.search(pattern, st)
        #     if pos_st:
        #         p_start = pos_st.start()
        #         p_end = pos_st.end()
        #         if p_start > 0:
        #             res += seg_and_tag(st[:p_start])
        #         st_num = st[p_start:p_end]
        #         if st_num[-2:] == ".0":
        #             st_num = st_num[:-2]
        #         if "." in st_num and st_num[-1] == "0":
        #             st_num = st_num[:-1]
        #         if nums.count(st_num) == 1:
        #             res.append("N"+str(nums.index(st_num)))
        #         else:
        #             res.append(st_num)
        #         if p_end < len(st):
        #             res += seg_and_tag(st[p_end:])
        #     else:
        #         for sst in st:
        #             res.append(sst)
        #     return res
        # out_seq = seg_and_tag(equations)

        # for s in out_seq:  # tag the num which is generated
        #     if s[0].isdigit() and s not in generate_nums and s not in nums:
        #         generate_nums.append(s)
        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
        if len(nums) != 0:
            pairs[key] = (input_seq, eq_segs, nums, num_pos)

    temp_g = []
    for g in generate_nums:
        if generate_nums[g] >= 5:
            temp_g.append(g)

    return pairs, temp_g, copy_nums


# Return a list of indexes, one for each word in the sentence, plus EOS
def indexes_from_sentence(lang, sentence, tree=False):
    res = []
    for word in sentence:
        if len(word) == 0:
            continue
        if word in lang.word2index:
            res.append(lang.word2index[word])
        else:
            res.append(lang.word2index["UNK"])
    if "EOS" in lang.index2word and not tree:
        res.append(lang.word2index["EOS"])
    return res
def indexes_to_sentence(lang, index_list, tree=False):
    res = []
    for index in index_list:
        if index < lang.n_words:
            res.append(lang.index2word[index])
    return res

def get_file_dict_vocab_by_file():
    file_dict_vocab={}
    file1=open("hownet//hownet_dict_vocab").readlines()
    print(file1[0])
    for x in file1:
        x_list=x.strip().split("###")
        word=x_list[0].encode('utf-8')
        word_list=[y.encode('utf-8') for y in x_list[1].split(" ")]
        file_dict_vocab[word]=word_list
    return file_dict_vocab
'''
def get_edge_matrix(hownet_dict_vocab,input_list):
    input_edge=[]
    for i in range(len(input_list)):
        temp_list=[]
        for j in range(len(input_list)):
            temp_list.append(0)
        input_edge.append(temp_list)
    for i in range(len(input_list)):
        word1 = input_list[i]
        if word1 in hownet_dict_vocab:
            cate1 = hownet_dict_vocab[word1]
            if len(cate1) >0:
                for j in range(len(input_list)):
                    word2= input_list[j]
                    if word2 in cate1:
                        input_edge[i][j]=1
    return input_edge
'''
def get_common_word(input_list):
    input_edge=[]
    for i in range(len(input_list)):
        temp_list=[]
        for j in range(len(input_list)):
            temp_list.append(0)
        input_edge.append(temp_list)
    for i in range(len(input_list)):
        word1 = input_list[i]
        for j in range(len(input_list)):
            word2= input_list[j]
            if i==j:
                input_edge[i][j]=1
            elif len(word1)>3 and word1!="NUM":
                if word2 in word1 or word2==word1:
                    input_edge[i][j]=1                
    return input_edge

def time_since(s):  # compute time
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)

def generate_how_dict_vocab(lang):
    hownet_dict_vocab={}
    hownet_dict_all={}
    hownet_dict_category={}
    vocab_list=[]
    uselese_tag=["属性值","文字","属性","ProperName|专","surname|姓","部件","人","human|人","time|时间"]
    index_=0
    start = time.time()
    file1=open("hownet//hownet_dict_all").readlines()
    for x in file1:
        x_list=x.strip().split("###")
        word=x_list[0].encode('utf-8')
        if len(word) > 3:
            #print(word)
            #print(len(word))
            word_list=[]
            if len(x_list[1])!=0:
                for y in x_list[1].strip().split(" "):
                    y=y.encode('utf-8')
                    if y not in uselese_tag and len(y)>0:
                        word_list.append(y)
            hownet_dict_all[word]=word_list
            vocab_list.append(word)
            for cate_ in word_list:
                if cate_ in hownet_dict_category:
                    category_list=hownet_dict_category[cate_]
                    if word not in category_list:
                        hownet_dict_category[cate_].append(word)
                else:
                    category_list=[]
                    category_list.append(word)
                    hownet_dict_category[cate_]=category_list
    print(hownet_dict_all["电话线"])
    start = time.time()
    count_all=0
    hownet_dict_tag={}
    for i in range(len(vocab_list)):
        word1=vocab_list[i]
        cate1=hownet_dict_all[word1]
        if len(cate1)==0:
            empty_list=[]
            hownet_dict_vocab[word1]=empty_list
        else:
            for word_ in cate1:
                if len(word_) >0 and len(word1)>0 and word1 !=None:

                    if word_ not in hownet_dict_tag:
                        empty_list=[]
                        hownet_dict_tag[word_]=empty_list
                    word_list=hownet_dict_tag[word_]
                    if word1 not in word_list:
                        word_list.append(word1)
                        hownet_dict_tag[word_]=word_list

            connect_word=[]
            for j in range(len(vocab_list)):
                if i!=j:
                    word2=vocab_list[j]
                    cate2=hownet_dict_all[word2]
                    flag=0
                    for word_ in cate1:
                        if word_ in cate2:
                            #print(word_+"#"+word1+"#"+word2)
                            flag=1
                            break
                    if flag==1:
                        count_all+=1
                        connect_word.append(word2)
            hownet_dict_vocab[word1]=connect_word

    print("training time", time_since(time.time() - start))
    print(len(vocab_list))
    print(count_all)
    output=open("hownet//hownet_dict_vocab","w")
    for word in hownet_dict_vocab:
        output.write(word+"###"+" ".join(hownet_dict_vocab[word])+"\n")
    output=open("hownet//hownet_dict_tag","w")
    for word in hownet_dict_tag:
        output.write(word+"###"+" ".join(hownet_dict_tag[word])+"\n")

    output=open("hownet//hownet_dict_category","w")
    output1=open("hownet//hownet_category_vocab","w")
    category_vocab=[]
    category_vocab.append("PAD")
    output1.write("PAD"+"\n")
    for word in hownet_dict_category:
        output.write(word+"###"+" ".join(hownet_dict_category[word])+"\n")
        category_vocab.append(word)
        output1.write(word+"\n")

    return hownet_dict_all,hownet_dict_category,category_vocab

def get_file_dict_vocab_by_file():
    file_dict_vocab={}
    file1=open("hownet//hownet_dict_vocab").readlines()
    print(file1[0])
    for x in file1:
        x_list=x.strip().split("###")
        word=x_list[0].encode('utf-8')
        word_list=[y.encode('utf-8') for y in x_list[1].split(" ")]
        file_dict_vocab[word]=word_list
    return file_dict_vocab
def get_edge_matrix(hownet_dict_vocab,input_list):
    input_edge=[]
    for i in range(len(input_list)):
        temp_list=[]
        for j in range(len(input_list)):
            temp_list.append(0)
        input_edge.append(temp_list)
    for i in range(len(input_list)):
        word1 = input_list[i]
        input_edge[i][i]=1
        #if i>0:
        #    input_edge[i][i-1]=1
        #if i<len(input_list)-1:
        #    input_edge[i][i+1]=1
        if word1 in hownet_dict_vocab:
            cate1 = hownet_dict_vocab[word1]
            if len(cate1) >0:
                for j in range(len(input_list)):
                    word2= input_list[j]
                    #if word2 in word1
                    if word2==word1 and len(word1)>3 and word1!="NUM":
                        input_edge[i][j]=1
                        input_edge[j][i]=1
                    '''
                    elif word2 in word1 and len(word1)>3 and word1!="NUM":
                        input_edge[i][j]=1
                        input_edge[j][i]=1
                    elif word1 in word2 and len(word2)>3 and word2!="NUM":
                        input_edge[i][j]=1
                        input_edge[j][i]=1

                    if word2 in cate1:
                        input_edge[i][j]=1
                        input_edge[j][i]=1
                    '''
    return input_edge



def get_unit_vocab(filename):
    x_idx_to_word=[]
    x_word_to_idx={}
    encode_vocab_dataset = open(filename).readlines()
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

unit_word_list, unit_syno_to_word=get_unit_vocab("Unit//unit_vocabulary+-")
not_unit_word_list, not_unit_syno_to_word=get_unit_vocab("Unit//not_unit_vocabulary")

def get_constraint_unit(input_list,num_pos):
    rule3_list=[]
    list_front=["的","占","了"]
    list_after=[",",".","，","。"]
    count_index=0
    for i in range(len(input_list)-1):
        if input_list[i]=="NUM":
            flag=0
            if input_list[i+1]=="倍":
                if num_pos[count_index]==i:
                    rule3_list.append(count_index)
                else:
                    print("*************************")
                    print(input_list)
                    print(num_pos)
            elif i>0 and input_list[i-1] in list_front  and input_list[i+1] not in unit_word_list:
                if num_pos[count_index]==i:
                    rule3_list.append(count_index)
                else:
                    print("*************************")
                    print(input_list)
                    print(num_pos)

            count_index+=1

    unit_list=[]
    for i in range(len(input_list)):
        if input_list[i]=="NUM":
            str_temp=" ".join(input_list[i+1:])
            flag=0
            for word in unit_word_list:
                word1=word+" "
                if str_temp.startswith(word1):
                    flag=1
                    unit_temp=unit_syno_to_word[word]
                    unit_list.append(unit_temp)
                    break
            if flag==0:
                unit_temp=""
                unit_list.append(unit_temp)
    #print(" ".join(input_list))
    #print(rule3_list)
    #print("###".join(unit_list))
    if len(unit_list)!=len(num_pos):
        print("*************************")
        print(input_list)
        print(num_pos)


    '''
    #not_unit 的 倍
    for word in not_unit_word_list:
        word1=word+" "
        if str_temp.startswith(word1) or str_temp==word:
            unit_temp=not_unit_syno_to_word[word]
            break
    if unit_temp=="__UNK__" and str_temp!="":
        not_solve_flag+=n1+" "
    str_number_unit_list.append(n1+"***"+number1+"***"+"__UNK__")
    '''
    #rule3_list [1,2] whether this pos is a multiple NUM
    #unit_list  [排,个,排] unit for each num
    return unit_list,rule3_list
def get_middle_exp(output_list):
    operator=["+", "-","*", "/", "^"]
    middle_exp=[]
    for exp in output_list:
        if exp in operator:
            list_exp=[exp]
        else:
            list_exp=[exp,exp,exp]
        for i in range(len(middle_exp)-1,-1,-1):
            curr_list=middle_exp[i]
            if curr_list[0] in operator:
                if len(curr_list)<3:
                    middle_exp[i].append(exp)
                    break
        middle_exp.append(list_exp)

    assert len(middle_exp) == len(output_list)
    return middle_exp
def indexes_from_middle_output(lang, sentence, tree=False):
    res = []
    for word_list in sentence:
        temp_res=[]
        if len(word_list)==2:
            word_list.append(word_list[-1])

        if len(word_list)==1:
            word_list.append(word_list[0])
            word_list.append(word_list[0])
        for word in word_list:
            if word in lang.word2index:
                temp_res.append(lang.word2index[word])
            else:
                temp_res.append(5)
        if len(word_list)!=3:
            print("******************************")
            print(sentence)
            print("******************************")    
        res.append(temp_res)
    return res

def get_category_list(hownet_dict_all,hownet_dict_category,category_vocab,input_list):
    category_name_list=[] #cate_name
    category_index_list=[]#cate_index
    category_match_list=[]#cate_match_pos
    category_match_word_list=[]

    punc_list=[",","：","；","？","！","，","“","”",",",".","?","，","。","？","．","；","｡"]
    for i in range(len(input_list)):
        for j in range(len(input_list)):
            word1 = input_list[i]
            word2 = input_list[j]
            if word1 != word2:
                if word1 in hownet_dict_all and word2 in hownet_dict_all and word1 not in punc_list and word1!="NUM" and word2 not in punc_list and word2!="NUM":
                    cate1 = hownet_dict_all[word1]
                    cate2 = hownet_dict_all[word2]
                    if len(cate1) >0 and len(cate2) >0:
                        for cate_ in cate1:
                            if cate_ in cate2 and cate_ not in category_name_list:
                                category_name_list.append(cate_)
                                category_index_list.append(category_vocab.index(cate_))
                                match_temp=[]
                                match_word=""
                                category_word_temp=hownet_dict_category[cate_]
                                for k in range(len(input_list)):
                                    if input_list[k] in category_word_temp:
                                        match_temp.append(k)
                                        match_word+=input_list[k]+" "
                                category_match_list.append(match_temp)
                                category_match_word_list.append(match_word)
    return category_name_list,category_index_list,category_match_list,category_match_word_list



def prepare_data(pairs_trained, pairs_tested, trim_min_count, generate_nums, copy_nums, tree=False):
    input_lang = Lang()
    output_lang = Lang()
    train_pairs = []
    test_pairs = []

    print("Indexing words...")
    for pair in pairs_trained:
        if not tree:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
        elif pair[-1]:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
    input_lang.build_input_lang(trim_min_count)
    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)
    print(pairs_trained[0][0])
    #generate_how_dict_vocab(input_lang)
    #
    hownet_dict_all,hownet_dict_category,category_vocab=generate_how_dict_vocab(input_lang)
    
    hownet_dict_vocab=get_file_dict_vocab_by_file()
    for pair in pairs_trained:
        num_stack = []
        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])

        num_stack.reverse()
        middle_exp=get_middle_exp(pair[1])
        input_cell = indexes_from_sentence(input_lang, pair[0])
        output_cell = indexes_from_sentence(output_lang, pair[1], tree)
        middle_exp_cell= indexes_from_middle_output(output_lang, middle_exp, tree)
        input_edge=get_edge_matrix(hownet_dict_vocab,pair[0])
        unit_list,rule3_list=get_constraint_unit(pair[0],pair[3])
        category_name_list,category_index_list,category_match_list,category_match_word_list=get_category_list(hownet_dict_all,hownet_dict_category,category_vocab,pair[0])
        
        # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
        #                     pair[2], pair[3], num_stack, pair[4]))
        train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                            pair[2], pair[3], num_stack,category_index_list,category_match_list,middle_exp_cell,input_edge))
    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))
    for pair in pairs_tested:
        num_stack = []
        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])

        num_stack.reverse()
        middle_exp=get_middle_exp(pair[1])
        input_cell = indexes_from_sentence(input_lang, pair[0])
        output_cell = indexes_from_sentence(output_lang, pair[1], tree)
        middle_exp_cell= indexes_from_middle_output(output_lang, middle_exp, tree)
        input_edge=get_edge_matrix(hownet_dict_vocab,pair[0])
        category_name_list,category_index_list,category_match_list,category_match_word_list=get_category_list(hownet_dict_all,hownet_dict_category,category_vocab,pair[0])
        
        unit_list,rule3_list=get_constraint_unit(pair[0],pair[3])
        # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
        #                     pair[2], pair[3], num_stack, pair[4]))
        test_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                           pair[2], pair[3], num_stack,category_index_list,category_match_list,middle_exp_cell,input_edge))
    print('Number of testind data %d' % (len(test_pairs)))
    return input_lang, output_lang, train_pairs, test_pairs,category_vocab,hownet_dict_vocab


def prepare_de_data(pairs_trained, pairs_tested, trim_min_count, generate_nums, copy_nums, tree=False):
    input_lang = Lang()
    output_lang = Lang()
    train_pairs = []
    test_pairs = []

    print("Indexing words...")
    for pair in pairs_trained:
        input_lang.add_sen_to_vocab(pair[0])
        output_lang.add_sen_to_vocab(pair[1])

    input_lang.build_input_lang(trim_min_count)

    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)

    for pair in pairs_trained:
        num_stack = []
        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])

        num_stack.reverse()
        input_cell = indexes_from_sentence(input_lang, pair[0])
        # train_pairs.append([input_cell, len(input_cell), pair[1], 0, pair[2], pair[3], num_stack, pair[4]])
        train_pairs.append([input_cell, len(input_cell), pair[1], 0, pair[2], pair[3], num_stack])
    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))
    for pair in pairs_tested:
        num_stack = []
        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])

        num_stack.reverse()
        input_cell = indexes_from_sentence(input_lang, pair[0])
        output_cell = indexes_from_sentence(output_lang, pair[1], tree)
        # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
        #                     pair[2], pair[3], num_stack, pair[4]))
        test_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                           pair[2], pair[3], num_stack))
    print('Number of testind data %d' % (len(test_pairs)))
    # the following is to test out_equation
    # counter = 0
    # for pdx, p in enumerate(train_pairs):
    #     temp_out = allocation(p[2], 0.8)
    #     x = out_equation(p[2], p[4])
    #     y = out_equation(temp_out, p[4])
    #     if x != y:
    #         counter += 1
    #     ans = p[7]
    #     if ans[-1] == '%':
    #         ans = ans[:-1] + "/100"
    #     if "(" in ans:
    #         for idx, i in enumerate(ans):
    #             if i != "(":
    #                 continue
    #             else:
    #                 break
    #         ans = ans[:idx] + "+" + ans[idx:]
    #     try:
    #         if abs(eval(y + "-(" + x + ")")) < 1e-4:
    #             z = 1
    #         else:
    #             print(pdx, x, p[2], y, temp_out, eval(x), eval("(" + ans + ")"))
    #     except:
    #         print(pdx, x, p[2], y, temp_out, p[7])
    # print(counter)
    return input_lang, output_lang, train_pairs, test_pairs


# Pad a with the PAD symbol
def pad_seq(seq, seq_len, max_length):
    seq += [PAD_token for _ in range(max_length - seq_len)]
    return seq

def pad_input_edge(input_edge, seq_len, max_length):
    for i in range(len(input_edge)):
        input_edge[i]+=[PAD_token for _ in range(max_length-seq_len)]
    for i in range(max_length-seq_len):
        temp_list=[PAD_token for _ in range(max_length)]
        input_edge.append(temp_list)
    return input_edge

def pad_middle_exp(seq, seq_len, max_length):
    for _ in range(max_length - seq_len):
        pad_list=[PAD_token,PAD_token,PAD_token]
        seq.append(pad_list)
    return seq
def pad_input_edge(input_edge, seq_len, max_length):
    for i in range(len(input_edge)):
        input_edge[i]+=[PAD_token for _ in range(max_length-seq_len)]
    for i in range(max_length-seq_len):
        temp_list=[PAD_token for _ in range(max_length)]
        input_edge.append(temp_list)
    return input_edge

# prepare the batches
def prepare_train_batch(pairs_to_batch, batch_size):
    pairs = copy.deepcopy(pairs_to_batch)
    random.shuffle(pairs)  # shuffle the pairs
    pos = 0
    input_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    output_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    num_size_batches = []
    input_edge_batches = []
    rule3_list_batches=[]
    unit_list_batches=[]
    output_middle_batches=[]
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp[1], reverse=True)
        input_length = []
        output_length = []
        for _, i, _, j, _, _, _,_,_,_,_ in batch:
            input_length.append(i)
            output_length.append(j)
        input_lengths.append(input_length)
        output_lengths.append(output_length)
        input_len_max = input_length[0]
        output_len_max = max(output_length)
        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        num_size_batch = []
        input_edge_batch = []
        unit_list_batch=[]
        rule3_list_batch=[]
        output_middle_batch=[]
        for i, li, j, lj, num, num_pos, num_stack,unit_list,rule3_list,middle_exp_cell,input_edge in batch:
            num_batch.append(len(num))
            input_batch.append(pad_seq(i, li, input_len_max))
            output_batch.append(pad_seq(j, lj, output_len_max))
            num_stack_batch.append(num_stack)
            num_pos_batch.append(num_pos)
            num_size_batch.append(len(num_pos))
            unit_list_batch.append(unit_list)
            input_edge_batch.append(pad_input_edge(input_edge, li, input_len_max))
            #input_edge_batch.append(input_edge)
            rule3_list_batch.append(rule3_list)
            output_middle_batch.append(pad_middle_exp(middle_exp_cell,lj,output_len_max))
        input_batches.append(input_batch)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
        num_size_batches.append(num_size_batch)
        input_edge_batches.append(input_edge_batch)
        unit_list_batches.append(unit_list_batch)
        rule3_list_batches.append(rule3_list_batch)
        output_middle_batches.append(output_middle_batch)
    return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches,unit_list_batches,rule3_list_batches,output_middle_batches,input_edge_batches

def prepare_test_batch(pairs_to_batch, batch_size):
    pairs = copy.deepcopy(pairs_to_batch)
    pos = 0
    input_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    output_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    num_size_batches = []
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp[1], reverse=True)
        input_length = []
        output_length = []
        for _, i, _, j, _, _, _ in batch:
            input_length.append(i)
            output_length.append(j)
        input_lengths.append(input_length)
        output_lengths.append(output_length)
        input_len_max = input_length[0]
        output_len_max = max(output_length)
        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        num_size_batch = []
        for i, li, j, lj, num, num_pos, num_stack in batch:
            num_batch.append(num)
            input_batch.append(pad_seq(i, li, input_len_max))
            output_batch.append(pad_seq(j, lj, output_len_max))
            num_stack_batch.append(num_stack)
            num_pos_batch.append(num_pos)
            num_size_batch.append(len(num_pos))
        input_batches.append(input_batch)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
        num_size_batches.append(num_size_batch)
    return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches

def get_num_stack(eq, output_lang, num_pos):
    num_stack = []
    for word in eq:
        temp_num = []
        flag_not = True
        if word not in output_lang.index2word:
            flag_not = False
            for i, j in enumerate(num_pos):
                if j == word:
                    temp_num.append(i)
        if not flag_not and len(temp_num) != 0:
            num_stack.append(temp_num)
        if not flag_not and len(temp_num) == 0:
            num_stack.append([_ for _ in range(len(num_pos))])
    num_stack.reverse()
    return num_stack


def prepare_de_train_batch(pairs_to_batch, batch_size, output_lang, rate, english=False):
    pairs = []
    b_pairs = copy.deepcopy(pairs_to_batch)
    for pair in b_pairs:
        p = copy.deepcopy(pair)
        pair[2] = check_bracket(pair[2], english)

        temp_out = exchange(pair[2], rate)
        temp_out = check_bracket(temp_out, english)

        p[2] = indexes_from_sentence(output_lang, pair[2])
        p[3] = len(p[2])
        pairs.append(p)

        temp_out_a = allocation(pair[2], rate)
        temp_out_a = check_bracket(temp_out_a, english)

        if temp_out_a != pair[2]:
            p = copy.deepcopy(pair)
            p[6] = get_num_stack(temp_out_a, output_lang, p[4])
            p[2] = indexes_from_sentence(output_lang, temp_out_a)
            p[3] = len(p[2])
            pairs.append(p)

        if temp_out != pair[2]:
            p = copy.deepcopy(pair)
            p[6] = get_num_stack(temp_out, output_lang, p[4])
            p[2] = indexes_from_sentence(output_lang, temp_out)
            p[3] = len(p[2])
            pairs.append(p)

            if temp_out_a != pair[2]:
                p = copy.deepcopy(pair)
                temp_out_a = allocation(temp_out, rate)
                temp_out_a = check_bracket(temp_out_a, english)
                if temp_out_a != temp_out:
                    p[6] = get_num_stack(temp_out_a, output_lang, p[4])
                    p[2] = indexes_from_sentence(output_lang, temp_out_a)
                    p[3] = len(p[2])
                    pairs.append(p)
    print("this epoch training data is", len(pairs))
    random.shuffle(pairs)  # shuffle the pairs
    pos = 0
    input_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    output_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp[1], reverse=True)
        input_length = []
        output_length = []
        for _, i, _, j, _, _, _ in batch:
            input_length.append(i)
            output_length.append(j)
        input_lengths.append(input_length)
        output_lengths.append(output_length)
        input_len_max = input_length[0]
        output_len_max = max(output_length)
        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        for i, li, j, lj, num, num_pos, num_stack in batch:
            num_batch.append(len(num))
            input_batch.append(pad_seq(i, li, input_len_max))
            output_batch.append(pad_seq(j, lj, output_len_max))
            num_stack_batch.append(num_stack)
            num_pos_batch.append(num_pos)
        input_batches.append(input_batch)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
    return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches


# Multiplication exchange rate
def exchange(ex_copy, rate):
    ex = copy.deepcopy(ex_copy)
    idx = 1
    while idx < len(ex):
        s = ex[idx]
        if (s == "*" or s == "+") and random.random() < rate:
            lidx = idx - 1
            ridx = idx + 1
            if s == "+":
                flag = 0
                while not (lidx == -1 or ((ex[lidx] == "+" or ex[lidx] == "-") and flag == 0) or flag == 1):
                    if ex[lidx] == ")" or ex[lidx] == "]":
                        flag -= 1
                    elif ex[lidx] == "(" or ex[lidx] == "[":
                        flag += 1
                    lidx -= 1
                if flag == 1:
                    lidx += 2
                else:
                    lidx += 1

                flag = 0
                while not (ridx == len(ex) or ((ex[ridx] == "+" or ex[ridx] == "-") and flag == 0) or flag == -1):
                    if ex[ridx] == ")" or ex[ridx] == "]":
                        flag -= 1
                    elif ex[ridx] == "(" or ex[ridx] == "[":
                        flag += 1
                    ridx += 1
                if flag == -1:
                    ridx -= 2
                else:
                    ridx -= 1
            else:
                flag = 0
                while not (lidx == -1
                           or ((ex[lidx] == "+" or ex[lidx] == "-" or ex[lidx] == "*" or ex[lidx] == "/") and flag == 0)
                           or flag == 1):
                    if ex[lidx] == ")" or ex[lidx] == "]":
                        flag -= 1
                    elif ex[lidx] == "(" or ex[lidx] == "[":
                        flag += 1
                    lidx -= 1
                if flag == 1:
                    lidx += 2
                else:
                    lidx += 1

                flag = 0
                while not (ridx == len(ex)
                           or ((ex[ridx] == "+" or ex[ridx] == "-" or ex[ridx] == "*" or ex[ridx] == "/") and flag == 0)
                           or flag == -1):
                    if ex[ridx] == ")" or ex[ridx] == "]":
                        flag -= 1
                    elif ex[ridx] == "(" or ex[ridx] == "[":
                        flag += 1
                    ridx += 1
                if flag == -1:
                    ridx -= 2
                else:
                    ridx -= 1
            if lidx > 0 and ((s == "+" and ex[lidx - 1] == "-") or (s == "*" and ex[lidx - 1] == "/")):
                lidx -= 1
                ex = ex[:lidx] + ex[idx:ridx + 1] + ex[lidx:idx] + ex[ridx + 1:]
            else:
                ex = ex[:lidx] + ex[idx + 1:ridx + 1] + [s] + ex[lidx:idx] + ex[ridx + 1:]
            idx = ridx
        idx += 1
    return ex


def check_bracket(x, english=False):
    if english:
        for idx, s in enumerate(x):
            if s == '[':
                x[idx] = '('
            elif s == '}':
                x[idx] = ')'
        s = x[0]
        idx = 0
        if s == "(":
            flag = 1
            temp_idx = idx + 1
            while flag > 0 and temp_idx < len(x):
                if x[temp_idx] == ")":
                    flag -= 1
                elif x[temp_idx] == "(":
                    flag += 1
                temp_idx += 1
            if temp_idx == len(x):
                x = x[idx + 1:temp_idx - 1]
            elif x[temp_idx] != "*" and x[temp_idx] != "/":
                x = x[idx + 1:temp_idx - 1] + x[temp_idx:]
        while True:
            y = len(x)
            for idx, s in enumerate(x):
                if s == "+" and idx + 1 < len(x) and x[idx + 1] == "(":
                    flag = 1
                    temp_idx = idx + 2
                    while flag > 0 and temp_idx < len(x):
                        if x[temp_idx] == ")":
                            flag -= 1
                        elif x[temp_idx] == "(":
                            flag += 1
                        temp_idx += 1
                    if temp_idx == len(x):
                        x = x[:idx + 1] + x[idx + 2:temp_idx - 1]
                        break
                    elif x[temp_idx] != "*" and x[temp_idx] != "/":
                        x = x[:idx + 1] + x[idx + 2:temp_idx - 1] + x[temp_idx:]
                        break
            if y == len(x):
                break
        return x

    lx = len(x)
    for idx, s in enumerate(x):
        if s == "[":
            flag_b = 0
            flag = False
            temp_idx = idx
            while temp_idx < lx:
                if x[temp_idx] == "]":
                    flag_b += 1
                elif x[temp_idx] == "[":
                    flag_b -= 1
                if x[temp_idx] == "(" or x[temp_idx] == "[":
                    flag = True
                if x[temp_idx] == "]" and flag_b == 0:
                    break
                temp_idx += 1
            if not flag:
                x[idx] = "("
                x[temp_idx] = ")"
                continue
        if s == "(":
            flag_b = 0
            flag = False
            temp_idx = idx
            while temp_idx < lx:
                if x[temp_idx] == ")":
                    flag_b += 1
                elif x[temp_idx] == "(":
                    flag_b -= 1
                if x[temp_idx] == "[":
                    flag = True
                if x[temp_idx] == ")" and flag_b == 0:
                    break
                temp_idx += 1
            if not flag:
                x[idx] = "["
                x[temp_idx] = "]"
    return x


# Multiplication allocation rate
def allocation(ex_copy, rate):
    ex = copy.deepcopy(ex_copy)
    idx = 1
    lex = len(ex)
    while idx < len(ex):
        if (ex[idx] == "/" or ex[idx] == "*") and (ex[idx - 1] == "]" or ex[idx - 1] == ")"):
            ridx = idx + 1
            r_allo = []
            r_last = []
            flag = 0
            flag_mmd = False
            while ridx < lex:
                if ex[ridx] == "(" or ex[ridx] == "[":
                    flag += 1
                elif ex[ridx] == ")" or ex[ridx] == "]":
                    flag -= 1
                if flag == 0:
                    if ex[ridx] == "+" or ex[ridx] == "-":
                        r_last = ex[ridx:]
                        r_allo = ex[idx + 1: ridx]
                        break
                    elif ex[ridx] == "*" or ex[ridx] == "/":
                        flag_mmd = True
                        r_last = [")"] + ex[ridx:]
                        r_allo = ex[idx + 1: ridx]
                        break
                elif flag == -1:
                    r_last = ex[ridx:]
                    r_allo = ex[idx + 1: ridx]
                    break
                ridx += 1
            if len(r_allo) == 0:
                r_allo = ex[idx + 1:]
            flag = 0
            lidx = idx - 1
            flag_al = False
            flag_md = False
            while lidx > 0:
                if ex[lidx] == "(" or ex[lidx] == "[":
                    flag -= 1
                elif ex[lidx] == ")" or ex[lidx] == "]":
                    flag += 1
                if flag == 1:
                    if ex[lidx] == "+" or ex[lidx] == "-":
                        flag_al = True
                if flag == 0:
                    break
                lidx -= 1
            if lidx != 0 and ex[lidx - 1] == "/":
                flag_al = False
            if not flag_al:
                idx += 1
                continue
            elif random.random() < rate:
                temp_idx = lidx + 1
                temp_res = ex[:lidx]
                if flag_mmd:
                    temp_res += ["("]
                if lidx - 1 > 0:
                    if ex[lidx - 1] == "-" or ex[lidx - 1] == "*" or ex[lidx - 1] == "/":
                        flag_md = True
                        temp_res += ["("]
                flag = 0
                lidx += 1
                while temp_idx < idx - 1:
                    if ex[temp_idx] == "(" or ex[temp_idx] == "[":
                        flag -= 1
                    elif ex[temp_idx] == ")" or ex[temp_idx] == "]":
                        flag += 1
                    if flag == 0:
                        if ex[temp_idx] == "+" or ex[temp_idx] == "-":
                            temp_res += ex[lidx: temp_idx] + [ex[idx]] + r_allo + [ex[temp_idx]]
                            lidx = temp_idx + 1
                    temp_idx += 1
                temp_res += ex[lidx: temp_idx] + [ex[idx]] + r_allo
                if flag_md:
                    temp_res += [")"]
                temp_res += r_last
                return temp_res
        if ex[idx] == "*" and (ex[idx + 1] == "[" or ex[idx + 1] == "("):
            lidx = idx - 1
            l_allo = []
            temp_res = []
            flag = 0
            flag_md = False  # flag for x or /
            while lidx > 0:
                if ex[lidx] == "(" or ex[lidx] == "[":
                    flag += 1
                elif ex[lidx] == ")" or ex[lidx] == "]":
                    flag -= 1
                if flag == 0:
                    if ex[lidx] == "+":
                        temp_res = ex[:lidx + 1]
                        l_allo = ex[lidx + 1: idx]
                        break
                    elif ex[lidx] == "-":
                        flag_md = True  # flag for -
                        temp_res = ex[:lidx] + ["("]
                        l_allo = ex[lidx + 1: idx]
                        break
                elif flag == 1:
                    temp_res = ex[:lidx + 1]
                    l_allo = ex[lidx + 1: idx]
                    break
                lidx -= 1
            if len(l_allo) == 0:
                l_allo = ex[:idx]
            flag = 0
            ridx = idx + 1
            flag_al = False
            all_res = []
            while ridx < lex:
                if ex[ridx] == "(" or ex[ridx] == "[":
                    flag -= 1
                elif ex[ridx] == ")" or ex[ridx] == "]":
                    flag += 1
                if flag == 1:
                    if ex[ridx] == "+" or ex[ridx] == "-":
                        flag_al = True
                if flag == 0:
                    break
                ridx += 1
            if not flag_al:
                idx += 1
                continue
            elif random.random() < rate:
                temp_idx = idx + 1
                flag = 0
                lidx = temp_idx + 1
                while temp_idx < idx - 1:
                    if ex[temp_idx] == "(" or ex[temp_idx] == "[":
                        flag -= 1
                    elif ex[temp_idx] == ")" or ex[temp_idx] == "]":
                        flag += 1
                    if flag == 1:
                        if ex[temp_idx] == "+" or ex[temp_idx] == "-":
                            all_res += l_allo + [ex[idx]] + ex[lidx: temp_idx] + [ex[temp_idx]]
                            lidx = temp_idx + 1
                    if flag == 0:
                        break
                    temp_idx += 1
                if flag_md:
                    temp_res += all_res + [")"]
                elif ex[temp_idx + 1] == "*" or ex[temp_idx + 1] == "/":
                    temp_res += ["("] + all_res + [")"]
                temp_res += ex[temp_idx + 1:]
                return temp_res
        idx += 1
    return ex



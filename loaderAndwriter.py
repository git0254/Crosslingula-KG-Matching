import codecs
import numpy as np
import os

def load_word_embedding(embedding_path, word_idx):
    # 打开embedding文件
    with codecs.open(embedding_path, 'r', 'utf-8') as f:
        vecs = []
        # 遍历文件中的每一行
        for line in f:
            line = line.strip()
            # 如果该行只有两个元素，则跳过
            if len(line.split(" ")) == 2:
                continue
            info = line.split(' ')
            # 获取单词
            word = info[0]
            # 获取单词对应的向量
            vec = [float(v) for v in info[1:]]
            # 如果向量长度不为300，则跳过
            if len(vec) != 300:
                continue
            vecs.append(vec)
            # 将单词和对应的索引存入word_idx字典中
            word_idx[word] = len(word_idx.keys()) + 1  #  + 1 is due to that we already have an unknown word

    # 将向量列表转换为numpy数组
    return np.array(vecs)

# 定义一个函数，用于将单词索引写入文件
def write_word_idx(word_idx, path):
    # 获取文件路径中的目录部分
    dir = path[:path.rfind('/')]
    # 如果目录不存在，则创建目录
    if not os.path.exists(dir):
        os.makedirs(dir)

    # 打开文件，以写入模式
    with codecs.open(path, 'w', 'utf-8') as f:
        # 遍历单词索引
        for word in word_idx:
            # 将单词和对应的索引写入文件
            f.write(str(word)+" "+str(word_idx[word])+'\n')

def read_word_idx_from_file(path, if_key_is_int=False):
    # 定义一个空字典，用于存储单词和索引的对应关系
    word_idx = {}
    # 打开文件，以只读模式读取，编码格式为utf-8
    with codecs.open(path, 'r', 'utf-8') as f:
        # 读取文件的所有行
        lines = f.readlines()
        # 遍历每一行
        for line in lines:
            # 去除行首行尾的空格和换行符，并将行按空格分割成列表
            info = line.strip().split(" ")
            # 如果列表的长度不等于2，说明这一行只有一个单词
            if len(info) != 2:
                # 将空格的索引设置为列表中的第一个元素
                word_idx[' '] = int(info[0])
            else:
                # 如果if_key_is_int为True，说明单词的索引是整数
                if if_key_is_int:
                    # 将单词的索引设置为列表中的第一个元素，单词设置为列表中的第二个元素
                    word_idx[int(info[0])] = int(info[1])
                else:
                    # 否则，将单词的索引设置为列表中的第一个元素，单词设置为列表中的第二个元素
                    word_idx[info[0]] = int(info[1])
    # 返回字典
    return word_idx


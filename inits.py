import tensorflow as tf
import numpy as np

# DISCLAIMER:
# Parts of this code file are derived from
# https://github.com/tkipf/gcn
# which is under an identical MIT license as GraphSAGE

def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    # 生成一个均匀分布的随机数，范围为[-scale, scale]，数据类型为tf.float32
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    # 返回一个变量，初始值为initial，变量名为name
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    # 计算初始化范围
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    # 生成均匀分布的随机数
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    # 返回初始化后的变量
    return tf.Variable(initial, name=name)

def random(shape, name=None):
    # tf.get_variable('W_train',
    #                 shape=[self.word_vocab_size, self.word_embedding_dim],
                    # initializer=tf.contrib.layers.xavier_initializer()
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def zeros(shape, name=None):
    """All zeros."""
    # 创建一个全零的张量
    initial = tf.zeros(shape, dtype=tf.float32)
    # 将全零的张量转换为变量
    return tf.Variable(initial, name=name)

def ones(shape, name=None):
    """All ones."""
    # 创建一个全为1的张量
    initial = tf.ones(shape, dtype=tf.float32)
    # 返回一个变量，初始值为全为1的张量
    return tf.Variable(initial, name=name)

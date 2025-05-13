from layers import Layer
import tensorflow as tf

class UniformNeighborSampler(Layer):
    """
       Uniformly samples neighbors.
       Assumes that adj lists are padded with random re-sampling
    """
    def __init__(self, adj_info, **kwargs):
        # 调用父类的初始化方法
        super(UniformNeighborSampler, self).__init__(**kwargs)
        # 保存邻接信息
        self.adj_info = adj_info

    def _call(self, inputs):
        # 获取输入的ids和num_samples
        ids, num_samples = inputs
        # 使用tf.nn.embedding_lookup函数获取adj_info中对应ids的值
        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids)
        # 对adj_lists进行转置
        adj_lists = tf.transpose(tf.transpose(adj_lists))
        # 使用tf.slice函数对adj_lists进行切片，获取前num_samples个元素
        adj_lists = tf.slice(adj_lists, [0,0], [-1, num_samples])
        # 返回adj_lists
        return adj_lists

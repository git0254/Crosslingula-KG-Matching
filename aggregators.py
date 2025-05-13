import tensorflow as tf
from layers import Layer, Dense
from inits import glorot, zeros, random
from pooling import mean_pool
from match_utils import multi_highway_layer

#
class GatedMeanAggregator(Layer):#平均池化操作
    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
            dropout=0, bias=True, act=tf.nn.relu,
            name=None, concat=False, **kwargs):
        # 初始化函数，设置输入维度、输出维度、邻居输入维度、dropout、bias、激活函数、name、concat等参数
        super(GatedMeanAggregator, self).__init__(**kwargs)

        # 设置dropout、bias、激活函数、concat等参数
        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        # 如果name不为空，则添加斜杠
        if name is not None:
            name = '/' + name
        else:
            name = ''

        # 如果邻居输入维度为空，则设置为输入维度
        if neigh_input_dim == None:
            neigh_input_dim = input_dim

        # 如果concat为True，则输出维度为2倍输出维度
        if concat:
            self.output_dim = 2 * output_dim

        # 定义一个变量作用域，用于存储模型中的变量
        with tf.variable_scope(self.name + name + '_vars'):
            # 定义一个变量，用于存储邻居向量的投影矩阵
            self.vars['neigh_weights'] = glorot([neigh_input_dim, output_dim],  #neigh_weights邻居向量的投影矩阵
                                                name='neigh_weights')
            # 定义一个变量，用于存储自身向量的投影矩阵
            self.vars['self_weights'] = glorot([input_dim, output_dim],         #self_weights自身向量的投影矩阵
                                               name='self_weights')
            # 如果需要偏置，则定义一个变量，用于存储偏置
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

            # 定义一个变量，用于存储门控向量的投影矩阵
            self.vars['gate_weights'] = glorot([2*output_dim, 2*output_dim],
                                                name='gate_weights')
            # 定义一个变量，用于存储门控向量的偏置
            self.vars['gate_bias'] = zeros([2*output_dim], name='bias')


        # 设置输入维度
        self.input_dim = input_dim
        # 设置输出维度
        self.output_dim = output_dim

    def _call(self, inputs):
        # 获取输入的节点向量和邻居节点向量
        self_vecs, neigh_vecs = inputs

        # 对邻居节点向量和节点向量进行dropout操作
        neigh_vecs = tf.nn.dropout(neigh_vecs, 1-self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1-self.dropout)

        # 计算邻居节点的均值
        neigh_means = tf.reduce_mean(neigh_vecs, axis=1)

        # [nodes] x [out_dim]
        # 计算邻居节点向量和权重矩阵的乘积
        from_neighs = tf.matmul(neigh_means, self.vars['neigh_weights'])

        # 计算节点向量和权重矩阵的乘积
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])

        # 如果不进行concat操作，则将from_self和from_neighs相加
        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        # 如果进行concat操作，则将from_self和from_neighs拼接
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        # 如果有偏置，则将偏置加到输出上
        if self.bias:
            output += self.vars['bias']

        # 将self和neighs连接起来
        gate = tf.concat([from_self, from_neighs], axis=1)
        # 将连接后的结果乘以权重矩阵，并加上偏置
        gate = tf.matmul(gate, self.vars["gate_weights"]) + self.vars["gate_bias"]
        # 使用ReLU激活函数
        gate = tf.nn.relu(gate)

        # 返回激活后的结果乘以输出
        return gate*self.act(output)


class GCNAggregator(Layer):
    """
    Aggregates via mean followed by matmul and non-linearity.
    Same matmul parameters are used self vector and neighbor vectors.
    """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
                 dropout=0., bias=True, act=tf.nn.relu, name=None, concat=False, mode="train", **kwargs):
        # 初始化GCNAggregator类
        super(GCNAggregator, self).__init__(**kwargs)

        # 设置dropout率
        self.dropout = dropout
        # 设置是否添加偏置
        self.bias = bias
        # 设置激活函数
        self.act = act
        # 设置是否拼接输入
        self.concat = concat
        # 设置模式
        self.mode = mode

        # 如果没有指定邻居输入维度，则默认为输入维度
        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        # 如果指定了名称，则添加斜杠
        if name is not None:
            name = '/' + name
        else:
            name = ''

        # 在变量作用域中定义权重和偏置
        with tf.variable_scope(self.name + name + '_vars'):
            # 定义邻居权重
            self.vars['weights'] = glorot([neigh_input_dim, output_dim],
                                          name='neigh_weights')
            # 如果添加偏置，则定义偏置
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        # 如果开启了日志记录，则记录变量
        if self.logging:
            self._log_vars()

        # 设置输入和输出维度
        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):
        # 获取输入的节点向量和邻居节点向量
        self_vecs, neigh_vecs = inputs

        # 如果当前模式为训练模式，则对节点向量和邻居节点向量进行dropout操作
        if self.mode == "train":
            neigh_vecs = tf.nn.dropout(neigh_vecs, 1 - self.dropout)
            self_vecs = tf.nn.dropout(self_vecs, 1 - self.dropout)

        # 计算邻居节点向量和节点向量拼接后的均值
        means = tf.reduce_mean(tf.concat([neigh_vecs, tf.expand_dims(self_vecs, axis=1)], axis=1), axis=1)

        # [nodes] x [out_dim]
        # 将均值与权重矩阵相乘得到输出
        output = tf.matmul(means, self.vars['weights'])

        # 如果self.bias为True，则将self.vars['bias']加到output上
        if self.bias:
            output += self.vars['bias']

        # 返回self.act(output)和self.output_dim
        return self.act(output), self.output_dim

class MeanAggregator(Layer):
    """Aggregates via mean followed by matmul and non-linearity."""

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
            dropout=0, bias=True, act=tf.nn.relu,
            name=None, concat=False, mode="train", if_use_high_way=False, **kwargs):
        # 初始化MeanAggregator类
        super(MeanAggregator, self).__init__(**kwargs)

        # 设置dropout率
        self.dropout = dropout
        # 设置是否使用bias
        self.bias = bias
        # 设置激活函数
        self.act = act
        # 设置是否concat
        self.concat = concat
        # 设置模式
        self.mode = mode
        # 设置是否使用high way
        self.if_use_high_way = if_use_high_way

        # 如果name不为空，则添加斜杠
        if name is not None:
            name = '/' + name
        else:
            name = ''

        # 如果neigh_input_dim为空，则设置为input_dim
        if neigh_input_dim == None:
            neigh_input_dim = input_dim

        # 设置neigh_input_dim
        self.neigh_input_dim = neigh_input_dim

        # 如果concat为True，则output_dim为2 * output_dim
        if concat:
            self.output_dim = 2 * output_dim
        else:
            self.output_dim = output_dim

        # 在变量作用域中定义变量
        with tf.variable_scope(self.name + name + '_vars'):
            # 定义neigh_weights
            self.vars['neigh_weights'] = glorot([neigh_input_dim, output_dim],
                                                name='neigh_weights')
            # 定义self_weights
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                               name='self_weights')

            # self.vars['neigh_weights'] = random([neigh_input_dim, output_dim], name='neigh_weights')
            # self.vars['self_weights'] = random([input_dim, output_dim], name='neigh_weights')

            # 如果bias为True，则定义bias
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        # 设置input_dim
        self.input_dim = input_dim

        # 设置output_dim
        self.output_dim = output_dim

        # 如果concat为True，则output_dim为output_dim * 2
        if self.concat:
            self.output_dim = output_dim * 2

    def _call(self, inputs):
        # 获取输入的节点向量和邻居节点向量
        self_vecs, neigh_vecs = inputs

        # 如果当前模式为训练模式，则对节点向量和邻居节点向量进行dropout
        if self.mode == "train":
            neigh_vecs = tf.nn.dropout(neigh_vecs, 1-self.dropout)
            self_vecs = tf.nn.dropout(self_vecs, 1-self.dropout)

        # reduce_mean performs better than mean_pool
        neigh_means = tf.reduce_mean(neigh_vecs, axis=1)
        # neigh_means = mean_pool(neigh_vecs, neigh_len)

        # [nodes] x [out_dim]
        from_neighs = tf.matmul(neigh_means, self.vars['neigh_weights'])

        # 如果使用highway层，则使用multi_highway_layer函数
        if self.if_use_high_way:
            with tf.variable_scope("fw_hidden_highway"):
                fw_hidden = multi_highway_layer(from_neighs, self.neigh_input_dim, 1)

        # 计算self_vecs和self_weights的矩阵乘法
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])

        # 如果不使用concat，则将from_self和from_neighs相加
        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        # 如果使用concat，则将from_self和from_neighs在axis=1上拼接
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output), self.output_dim

class AttentionAggregator(Layer):
    """ Attention-based aggregator """
    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
            dropout=0, bias=True, act=tf.nn.relu,
            name=None, concat=False, mode="train", **kwargs):
        # 初始化函数，用于初始化AttentionAggregator类
        super(AttentionAggregator, self).__init__(**kwargs)

        # 设置dropout参数
        self.dropout = dropout
        # 设置bias参数
        self.bias = bias
        # 设置激活函数
        self.act = act
        # 设置是否concat参数
        self.concat = concat
        # 设置模式参数
        self.mode = mode

        # 如果name不为空，则添加斜杠
        if name is not None:
            name = '/' + name
        else:
            name = ''

        # 如果neigh_input_dim为空，则设置为input_dim
        if neigh_input_dim == None:
            neigh_input_dim = input_dim

        # 设置输入维度
        self.input_dim = input_dim
        # 设置输出维度
        self.output_dim = output_dim

        # 在变量作用域内定义变量
        with tf.variable_scope(self.name + name + '_vars'):
            # 如果bias为True，则定义bias变量
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

            # 定义q_dense_layer变量
            self.q_dense_layer = Dense(input_dim=input_dim, output_dim=input_dim, bias=False, sparse_inputs=False,
                                       name="q")
            # 定义k_dense_layer变量
            self.k_dense_layer = Dense(input_dim=input_dim, output_dim=input_dim, bias=False, sparse_inputs=False,
                                       name="k")
            # 定义v_dense_layer变量
            self.v_dense_layer = Dense(input_dim=input_dim, output_dim=input_dim, bias=False, sparse_inputs=False,
                                       name="v")

            # 定义output_dense_layer变量
            self.output_dense_layer = Dense(input_dim=input_dim, output_dim=output_dim, bias=False, sparse_inputs=False, name="output_transform")

    def _call(self, inputs):
        # 获取输入的self_vecs和neigh_vecs
        self_vecs, neigh_vecs= inputs

        # 对self_vecs进行线性变换
        q = self.q_dense_layer(self_vecs)

        # 将self_vecs和neigh_vecs进行拼接
        neigh_vecs = tf.concat([tf.expand_dims(self_vecs, axis=1), neigh_vecs], axis=1)
        # 获取neigh_vecs的长度
        neigh_len = tf.shape(neigh_vecs)[1]
        # 将neigh_vecs进行reshape
        neigh_vecs = tf.reshape(neigh_vecs, [-1, self.input_dim])

        # 对neigh_vecs进行线性变换
        k = self.k_dense_layer(neigh_vecs)
        v = self.v_dense_layer(neigh_vecs)

        # 将k和v进行reshape
        k = tf.reshape(k, [-1, neigh_len, self.input_dim])
        v = tf.reshape(v, [-1, neigh_len, self.input_dim])

        # 计算q和k的点积
        logits = tf.reduce_sum(tf.multiply(tf.expand_dims(q, axis=1), k), axis=-1)
        # 如果有bias，则加上bias
        # if self.bias:
        #     logits += self.vars['bias']

        # 计算softmax
        weights = tf.nn.softmax(logits, name="attention_weights")

        # 计算attention_output
        attention_output = tf.reduce_sum(tf.multiply(tf.expand_dims(weights, axis=-1), v), axis=1)

        # 对attention_output进行线性变换
        attention_output = self.output_dense_layer(attention_output)

        # 返回attention_output和output_dim
        return attention_output, self.output_dim

class MaxPoolingAggregator(Layer):
    """ Aggregates via max-pooling over MLP functions."""
    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
            dropout=0., bias=True, act=tf.nn.relu, name=None, concat=False, mode="train", **kwargs):
        super(MaxPoolingAggregator, self).__init__(**kwargs)

        # 设置模式
        self.mode = mode
        # 设置dropout
        self.dropout = dropout
        # 设置bias
        self.bias = bias
        # 设置激活函数
        self.act = act
        # 设置是否concat
        self.concat = concat

        # 如果name不为空，则添加斜杠
        if name is not None:
            name = '/' + name
        else:
            name = ''

        # 如果neigh_input_dim为空，则设置为input_dim
        if neigh_input_dim == None:
            neigh_input_dim = input_dim

        # 如果concat为True，则output_dim设置为2 * output_dim
        if concat:
            self.output_dim = 2 * output_dim

        # 如果model_size为"small"，则hidden_dim设置为50
        if model_size == "small":
            hidden_dim = self.hidden_dim = 50
        # 如果model_size为"big"，则hidden_dim设置为50
        elif model_size == "big":
            hidden_dim = self.hidden_dim = 50

        # 初始化mlp_layers
        self.mlp_layers = []
        # 添加一个Dense层
        self.mlp_layers.append(Dense(input_dim=neigh_input_dim, output_dim=hidden_dim, act=tf.nn.relu,
                                     dropout=dropout, sparse_inputs=False, logging=self.logging))

        # 在variable_scope中定义变量
        with tf.variable_scope(self.name + name + '_vars'):

            # 定义neigh_weights
            self.vars['neigh_weights'] = glorot([hidden_dim, output_dim], name='neigh_weights')

            # 定义self_weights
            self.vars['self_weights'] = glorot([input_dim, output_dim], name='self_weights')

            # 如果bias为True，则定义bias
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        # 设置input_dim
        self.input_dim = input_dim
        # 设置output_dim
        self.output_dim = output_dim
        # 设置neigh_input_dim
        self.neigh_input_dim = neigh_input_dim

        # 如果concat为True，则output_dim设置为output_dim * 2
        if self.concat:
            self.output_dim = output_dim * 2

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs
        neigh_h = neigh_vecs

        dims = tf.shape(neigh_h)
        batch_size = dims[0]
        num_neighbors = dims[1]

        h_reshaped = tf.reshape(neigh_h, (batch_size * num_neighbors, self.neigh_input_dim))

        for l in self.mlp_layers:
            h_reshaped = l(h_reshaped)
        neigh_h = tf.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim))
        neigh_h = tf.reduce_max(neigh_h, axis=1)

        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output), self.output_dim


class SeqAggregator(Layer):
    """ Aggregates via a standard LSTM.
    """

    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
                 dropout=0., bias=True, act=tf.nn.relu, name=None, concat=False, mode="train", **kwargs):
        super(SeqAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat
        self.mode = mode
        self.output_dim = output_dim

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if model_size == "small":
            hidden_dim = self.hidden_dim = 128
        elif model_size == "big":
            hidden_dim = self.hidden_dim = 256

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([hidden_dim, output_dim],
                                                name='neigh_weights')

            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                               name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim
        self.cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim)

    def _call(self, inputs):
        # 获取输入的节点向量和邻居节点的向量
        self_vecs, neigh_vecs = inputs

        # 获取邻居节点的向量形状
        dims = tf.shape(neigh_vecs)
        # 获取批处理大小
        batch_size = dims[0]
        # 初始化RNN的初始状态
        initial_state = self.cell.zero_state(batch_size, tf.float32)
        # 计算邻居节点的向量中非零元素的个数
        used = tf.sign(tf.reduce_max(tf.abs(neigh_vecs), axis=2))
        # 计算邻居节点的向量长度
        length = tf.reduce_sum(used, axis=1)
        # 将长度设置为1
        length = tf.maximum(length, tf.constant(1.))
        # 将长度转换为整数
        length = tf.cast(length, tf.int32)

        # 定义RNN的变量作用域
        with tf.variable_scope(self.name) as scope:
            try:
                # 使用dynamic_rnn函数计算RNN的输出和状态
                rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                    self.cell, neigh_vecs,
                    initial_state=initial_state, dtype=tf.float32, time_major=False,
                    sequence_length=length)
            except ValueError:
                # 如果出现ValueError，则重用变量
                scope.reuse_variables()
                rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                    self.cell, neigh_vecs,
                    initial_state=initial_state, dtype=tf.float32, time_major=False,
                    sequence_length=length)
        # 获取RNN输出的批处理大小和最大长度
        batch_size = tf.shape(rnn_outputs)[0]
        max_len = tf.shape(rnn_outputs)[1]
        # 获取RNN输出的维度
        out_size = int(rnn_outputs.get_shape()[2])
        # 计算索引
        index = tf.range(0, batch_size) * max_len + (length - 1)
        # 将RNN输出展平
        flat = tf.reshape(rnn_outputs, [-1, out_size])
        # 根据索引获取邻居节点的向量
        neigh_h = tf.gather(flat, index)

        # 计算邻居节点的向量与邻居权重矩阵的乘积
        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        # 计算节点向量的与节点权重矩阵的乘积
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])

        # 将节点向量和邻居节点的向量相加
        output = tf.add_n([from_self, from_neighs])

        # 如果不进行拼接，则将节点向量和邻居节点的向量相加
        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        # 如果进行拼接，则将节点向量和邻居节点的向量拼接
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

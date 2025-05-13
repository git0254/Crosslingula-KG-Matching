import tensorflow as tf
from tensorflow.python.ops import nn_ops

def my_lstm_layer(input_reps, lstm_dim, input_lengths=None, scope_name=None, reuse=False, is_training=True,
                  dropout_rate=0.2, use_cudnn=True):
    '''
    :param inputs: [batch_size, seq_len, feature_dim]
    :param lstm_dim:
    :param scope_name:
    :param reuse:
    :param is_training:
    :param dropout_rate:
    :return:
    '''
    # 对输入进行dropout操作
    input_reps = dropout_layer(input_reps, dropout_rate, is_training=is_training)
    # 定义lstm层的变量范围
    with tf.variable_scope(scope_name, reuse=reuse):
        # 如果使用cudnn
        if use_cudnn:
            # 转置输入
            inputs = tf.transpose(input_reps, [1, 0, 2])
            # 定义cudnn的bidirectional lstm层
            lstm = tf.contrib.cudnn_rnn.CudnnLSTM(1, lstm_dim, direction="bidirectional",
                                    name="{}_cudnn_bi_lstm".format(scope_name), dropout=dropout_rate if is_training else 0)
            # 运行lstm层
            outputs, _ = lstm(inputs)
            # 转置输出
            outputs = tf.transpose(outputs, [1, 0, 2])
            # 获取前向和后向的输出
            f_rep = outputs[:, :, 0:lstm_dim]
            b_rep = outputs[:, :, lstm_dim:2*lstm_dim]
        # 如果不使用cudnn
        else:
            # 定义前向和后向的lstm cell
            context_lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(lstm_dim)
            context_lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(lstm_dim)
            # 如果是训练模式
            if is_training:
                # 定义前向lstm cell，并添加dropout
                context_lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(context_lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
                # 定义后向lstm cell，并添加dropout
                context_lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(context_lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
            # 定义多层的lstm cell
            context_lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([context_lstm_cell_fw])
            context_lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([context_lstm_cell_bw])

            # 运行bidirectional dynamic rnn
            (f_rep, b_rep), _ = tf.nn.bidirectional_dynamic_rnn(
                context_lstm_cell_fw, context_lstm_cell_bw, input_reps, dtype=tf.float32,
                sequence_length=input_lengths)  # [batch_size, question_len, context_lstm_dim]
            # 将前向和后向的输出拼接
            outputs = tf.concat(axis=2, values=[f_rep, b_rep])
    # 返回前向和后向的输出以及拼接后的输出
    return (f_rep,b_rep, outputs)

def dropout_layer(input_reps, dropout_rate, is_training=True):
    # 如果is_training为True，则使用dropout_rate对input_reps进行dropout操作
    if is_training:
        output_repr = tf.nn.dropout(input_reps, (1 - dropout_rate))
    # 如果is_training为False，则直接返回input_reps
    else:
        output_repr = input_reps
    # 返回dropout后的结果
    return output_repr

def cosine_distance(y1,y2, cosine_norm=True, eps=1e-6):
    # cosine_norm = True
    # y1 [....,a, 1, d]
    # y2 [....,1, b, d]
    # 计算y1和y2的点积
    cosine_numerator = tf.reduce_sum(tf.multiply(y1, y2), axis=-1)
    # 如果cosine_norm为False，则返回tanh(cosine_numerator)
    if not cosine_norm:
        return tf.tanh(cosine_numerator)
    # 计算y1的范数
    y1_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y1), axis=-1), eps))
    # 计算y2的范数
    y2_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y2), axis=-1), eps))
    # 返回cosine_numerator / y1_norm / y2_norm
    return cosine_numerator / y1_norm / y2_norm

# 定义欧几里得距离函数
def euclidean_distance(y1, y2, eps=1e-6):
    # 计算y1和y2之间的欧几里得距离
    distance = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y1 - y2), axis=-1), eps))
    # 返回欧几里得距离
    return distance

def cross_entropy(logits, truth, mask=None):
    # logits: [batch_size, passage_len]
    # truth: [batch_size, passage_len]
    # mask: [batch_size, passage_len]
    if mask is not None: logits = tf.multiply(logits, mask) # 如果mask不为空，则将logits与mask相乘
    xdev = tf.subtract(logits, tf.expand_dims(tf.reduce_max(logits, 1), -1)) # 计算logits与logits中每一行的最大值之差
    log_predictions = tf.subtract(xdev, tf.expand_dims(tf.log(tf.reduce_sum(tf.exp(xdev),-1)),-1)) # 计算log_predictions，即logits减去exp(xdev)的和的对数
    result = tf.multiply(truth, log_predictions) # [batch_size, passage_len]，将truth与log_predictions相乘
    if mask is not None: result = tf.multiply(result, mask) # [batch_size, passage_len]，如果mask不为空，则将result与mask相乘
    return tf.multiply(-1.0,tf.reduce_sum(result, -1)) # [batch_size]，将result的每一行的和乘以-1，并返回

def projection_layer(in_val, input_size, output_size, activation_func=tf.tanh, scope=None):
    # in_val: [batch_size, passage_len, dim]
    input_shape = tf.shape(in_val)
    batch_size = input_shape[0]
    passage_len = input_shape[1]
#     feat_dim = input_shape[2]
    # 将输入的维度重新排列，将batch_size和passage_len合并
    in_val = tf.reshape(in_val, [batch_size * passage_len, input_size])
    # 创建一个变量作用域
    with tf.variable_scope(scope or "projection_layer"):
        # 创建一个全连接层的权重矩阵
        full_w = tf.get_variable("full_w", [input_size, output_size], dtype=tf.float32)
        # 创建一个全连接层的偏置向量
        full_b = tf.get_variable("full_b", [output_size], dtype=tf.float32)
        # 使用全连接层进行线性变换，并使用激活函数
        outputs = activation_func(tf.nn.xw_plus_b(in_val, full_w, full_b))
    # 将输出的维度重新排列，恢复原来的batch_size和passage_len
    outputs = tf.reshape(outputs, [batch_size, passage_len, output_size])
    return outputs # [batch_size, passage_len, output_size]

def highway_layer(in_val, output_size, activation_func=tf.tanh, scope=None):
    # in_val: [batch_size, passage_len, dim]
    # 获取输入数据的形状
    input_shape = tf.shape(in_val)
    # 获取输入数据的batch_size
    batch_size = input_shape[0]
    # 获取输入数据的passage_len
    passage_len = input_shape[1]
#     feat_dim = input_shape[2]
    in_val = tf.reshape(in_val, [batch_size * passage_len, output_size])
    with tf.variable_scope(scope or "highway_layer"):
        # 定义highway层的权重和偏置
        highway_w = tf.get_variable("highway_w", [output_size, output_size], dtype=tf.float32)
        highway_b = tf.get_variable("highway_b", [output_size], dtype=tf.float32)
        full_w = tf.get_variable("full_w", [output_size, output_size], dtype=tf.float32)
        full_b = tf.get_variable("full_b", [output_size], dtype=tf.float32)
        # 计算全连接层的输出
        trans = activation_func(tf.nn.xw_plus_b(in_val, full_w, full_b))
        # 计算门控单元的输出
        gate = tf.nn.sigmoid(tf.nn.xw_plus_b(in_val, highway_w, highway_b))
        # 计算highway层的输出
        outputs = tf.add(tf.multiply(trans, gate), tf.multiply(in_val, tf.subtract(1.0, gate)), "y")
    outputs = tf.reshape(outputs, [batch_size, passage_len, output_size])
    return outputs

# 定义一个多层高速公路网络函数
def multi_highway_layer(in_val, output_size, num_layers, activation_func=tf.tanh, scope_name=None, reuse=False):
    # 在指定的作用域内创建变量
    with tf.variable_scope(scope_name, reuse=reuse):
        # 循环num_layers次
        for i in xrange(num_layers):
            # 定义当前作用域的名称
            cur_scope_name = scope_name + "-{}".format(i)
            # 调用高速公路网络函数
            in_val = highway_layer(in_val, output_size,activation_func=activation_func, scope=cur_scope_name)
    # 返回结果
    return in_val

def collect_representation(representation, positions):
    # representation: [batch_size, node_num, feature_dim]
    # positions: [batch_size, neigh_num]
    return collect_probs(representation, positions)

def collect_final_step_of_lstm(lstm_representation, lengths):
    # lstm_representation: [batch_size, passsage_length, dim]
    # lengths: [batch_size]
    lengths = tf.maximum(lengths, tf.zeros_like(lengths, dtype=tf.int32))

    # 获取batch_size
    batch_size = tf.shape(lengths)[0]
    # 创建一个从0到batch_size的序列
    batch_nums = tf.range(0, limit=batch_size) # shape (batch_size)
    # 将batch_nums和lengths组合成索引
    indices = tf.stack((batch_nums, lengths), axis=1) # shape (batch_size, 2)
    # 根据索引从lstm_representation中获取结果
    result = tf.gather_nd(lstm_representation, indices, name='last-forwar-lstm')
    return result # [batch_size, dim]

def collect_probs(probs, positions):
    # probs [batch_size, chunks_size]
    # positions [batch_size, pair_size]
    batch_size = tf.shape(probs)[0] # 获取probs的batch_size
    pair_size = tf.shape(positions)[1] # 获取positions的pair_size
    batch_nums = tf.range(0, limit=batch_size) # shape (batch_size)
    batch_nums = tf.reshape(batch_nums, shape=[-1, 1]) # [batch_size, 1]
    batch_nums = tf.tile(batch_nums, multiples=[1, pair_size]) # [batch_size, pair_size]

    # 创建一个索引矩阵，用于从probs中获取对应位置的值
    indices = tf.stack((batch_nums, positions), axis=2) # shape (batch_size, pair_size, 2)
    pair_probs = tf.gather_nd(probs, indices) # 根据索引从probs中获取对应位置的值
    # pair_probs = tf.reshape(pair_probs, shape=[batch_size, pair_size])
    return pair_probs


def calcuate_attention(in_value_1, in_value_2, feature_dim1, feature_dim2, scope_name='att',
                       att_type='symmetric', att_dim=20, remove_diagnoal=False, mask1=None, mask2=None, is_training=False, dropout_rate=0.2):
    # 获取输入值的形状
    input_shape = tf.shape(in_value_1)
    # 获取输入值的第一个维度，即batch_size
    batch_size = input_shape[0]
    # 获取输入值的第二个维度，即len_1
    len_1 = input_shape[1]
    # 获取输入值的第二个维度，即len_2
    len_2 = tf.shape(in_value_2)[1]

    # 对输入值in_value_1进行dropout操作
    in_value_1 = dropout_layer(in_value_1, dropout_rate, is_training=is_training)
    # 对输入值in_value_2进行dropout操作
    in_value_2 = dropout_layer(in_value_2, dropout_rate, is_training=is_training)
    with tf.variable_scope(scope_name):
        # calculate attention ==> a: [batch_size, len_1, len_2]
        # 定义注意力权重矩阵atten_w1
        atten_w1 = tf.get_variable("atten_w1", [feature_dim1, att_dim], dtype=tf.float32)
        # 如果两个特征的维度相同，则atten_w2等于atten_w1
        if feature_dim1 == feature_dim2: atten_w2 = atten_w1
        # 否则，定义注意力权重矩阵atten_w2
        else: atten_w2 = tf.get_variable("atten_w2", [feature_dim2, att_dim], dtype=tf.float32)
        # 计算第一个特征的注意力值atten_value_1
        atten_value_1 = tf.matmul(tf.reshape(in_value_1, [batch_size * len_1, feature_dim1]), atten_w1)  # [batch_size*len_1, feature_dim]
        # 将atten_value_1重新 reshape 为 [batch_size, len_1, att_dim]
        atten_value_1 = tf.reshape(atten_value_1, [batch_size, len_1, att_dim])
        # 计算第二个特征的注意力值atten_value_2
        atten_value_2 = tf.matmul(tf.reshape(in_value_2, [batch_size * len_2, feature_dim2]), atten_w2)  # [batch_size*len_2, feature_dim]
        # 将atten_value_2重新 reshape 为 [batch_size, len_2, att_dim]
        atten_value_2 = tf.reshape(atten_value_2, [batch_size, len_2, att_dim])


        if att_type == 'additive':
            # 创建一个可训练的变量atten_b，维度为att_dim
            atten_b = tf.get_variable("atten_b", [att_dim], dtype=tf.float32)
            # 创建一个可训练的变量atten_v，维度为1，att_dim
            atten_v = tf.get_variable("atten_v", [1, att_dim], dtype=tf.float32)
            atten_value_1 = tf.expand_dims(atten_value_1, axis=2, name="atten_value_1")  # [batch_size, len_1, 'x', feature_dim]
            atten_value_2 = tf.expand_dims(atten_value_2, axis=1, name="atten_value_2")  # [batch_size, 'x', len_2, feature_dim]
            atten_value = atten_value_1 + atten_value_2  # + tf.expand_dims(tf.expand_dims(tf.expand_dims(atten_b, axis=0), axis=0), axis=0)
            atten_value = nn_ops.bias_add(atten_value, atten_b)
            atten_value = tf.tanh(atten_value)  # [batch_size, len_1, len_2, feature_dim]
            atten_value = tf.reshape(atten_value, [-1, att_dim]) * atten_v  # tf.expand_dims(atten_v, axis=0) # [batch_size*len_1*len_2, feature_dim]
            atten_value = tf.reduce_sum(atten_value, axis=-1)
            atten_value = tf.reshape(atten_value, [batch_size, len_1, len_2])
        else:
            atten_value_1 = tf.tanh(atten_value_1)
            # atten_value_1 = tf.nn.relu(atten_value_1)
            atten_value_2 = tf.tanh(atten_value_2)
            # atten_value_2 = tf.nn.relu(atten_value_2)
            diagnoal_params = tf.get_variable("diagnoal_params", [1, 1, att_dim], dtype=tf.float32)
            atten_value_1 = atten_value_1 * diagnoal_params
            atten_value = tf.matmul(atten_value_1, atten_value_2, transpose_b=True) # [batch_size, len_1, len_2]

        # normalize
        if remove_diagnoal:
            diagnoal = tf.ones([len_1], tf.float32)  # [len1]
            diagnoal = 1.0 - tf.diag(diagnoal)  # [len1, len1]
            diagnoal = tf.expand_dims(diagnoal, axis=0)  # ['x', len1, len1]
            atten_value = atten_value * diagnoal
        if mask1 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask1, axis=-1))
        if mask2 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask2, axis=1))
        atten_value = tf.nn.softmax(atten_value, name='atten_value')  # [batch_size, len_1, len_2]
        if remove_diagnoal: atten_value = atten_value * diagnoal
        if mask1 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask1, axis=-1))
        if mask2 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask2, axis=1))

    return atten_value

def weighted_sum(atten_scores, in_values):
    '''

    :param atten_scores: # [batch_size, len1, len2]
    :param in_values: [batch_size, len2, dim]
    :return:
    '''
    return tf.matmul(atten_scores, in_values)

def cal_relevancy_matrix(in_question_repres, in_passage_repres):
    in_question_repres_tmp = tf.expand_dims(in_question_repres, 1) # [batch_size, 1, question_len, dim]
    in_passage_repres_tmp = tf.expand_dims(in_passage_repres, 2) # [batch_size, passage_len, 1, dim]
    relevancy_matrix = cosine_distance(in_question_repres_tmp,in_passage_repres_tmp) # [batch_size, passage_len, question_len]
    return relevancy_matrix

def mask_relevancy_matrix(relevancy_matrix, question_mask, passage_mask):
    # relevancy_matrix: [batch_size, passage_len, question_len]
    # question_mask: [batch_size, question_len]
    # passage_mask: [batch_size, passsage_len]
    if question_mask is not None:
        # 将relevancy_matrix与question_mask相乘，得到新的relevancy_matrix
        relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(question_mask, 1))
    # 将relevancy_matrix与passage_mask相乘，得到新的relevancy_matrix
    relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(passage_mask, 2))
    return relevancy_matrix

# 定义一个函数，用于计算给定张量和变量列表的梯度
def compute_gradients(tensor, var_list):
  # 使用tf.gradients函数计算给定张量和变量列表的梯度
  grads = tf.gradients(tensor, var_list)
  # 返回一个列表，其中包含每个变量的梯度，如果梯度为None，则返回该变量的零张量
  return [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(var_list, grads)]

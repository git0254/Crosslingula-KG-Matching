import tensorflow as tf
import layer_utils

eps = 1e-6


def cosine_distance(y1, y2):
    # y1 [....,a, 1, d]
    # y2 [....,1, b, d]
    # 计算y1和y2的点积
    cosine_numerator = tf.reduce_sum(tf.multiply(y1, y2), axis=-1)
    # 计算y1的范数
    y1_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y1), axis=-1), eps))
    # 计算y2的范数
    y2_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y2), axis=-1), eps))
    # 返回余弦距离
    return cosine_numerator / y1_norm / y2_norm


def cal_relevancy_matrix(in_question_repres, in_passage_repres):
    # 将in_question_repres扩展为[batch_size, 1, question_len, dim]的形状
    in_question_repres_tmp = tf.expand_dims(in_question_repres, 1)  # [batch_size, 1, question_len, dim]
    # 将in_passage_repres扩展为[batch_size, passage_len, 1, dim]的形状
    in_passage_repres_tmp = tf.expand_dims(in_passage_repres, 2)  # [batch_size, passage_len, 1, dim]
    # 计算in_question_repres_tmp和in_passage_repres_tmp之间的余弦距离，得到relevancy_matrix
    relevancy_matrix = cosine_distance(in_question_repres_tmp,
                                       in_passage_repres_tmp)  # [batch_size, passage_len, question_len]
    # 返回relevancy_matrix
    return relevancy_matrix


def mask_relevancy_matrix(relevancy_matrix, question_mask, passage_mask):
    # relevancy_matrix: [batch_size, passage_len, question_len]
    # question_mask: [batch_size, question_len]
    # passage_mask: [batch_size, passsage_len]
    # 将relevancy_matrix与question_mask进行逐元素相乘，得到新的relevancy_matrix
    relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(question_mask, 1))
    # 将relevancy_matrix与passage_mask进行逐元素相乘，得到最终的relevancy_matrix
    relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(passage_mask, 2))
    return relevancy_matrix


def multi_perspective_expand_for_3D(in_tensor, decompose_params):
    # 对输入张量进行扩展，增加一个维度
    in_tensor = tf.expand_dims(in_tensor, axis=2)  # [batch_size, passage_len, 'x', dim]
    # 对分解参数进行扩展，增加两个维度
    decompose_params = tf.expand_dims(tf.expand_dims(decompose_params, axis=0), axis=0)  # [1, 1, decompse_dim, dim]
    # 将输入张量和分解参数进行相乘
    return tf.multiply(in_tensor, decompose_params)  # [batch_size, passage_len, decompse_dim, dim]


def multi_perspective_expand_for_2D(in_tensor, decompose_params):
    # 对输入张量进行扩展，增加一个维度
    in_tensor = tf.expand_dims(in_tensor, axis=1)  # [batch_size, 'x', dim]
    # 对分解参数进行扩展，增加一个维度
    decompose_params = tf.expand_dims(decompose_params, axis=0)  # [1, decompse_dim, dim]
    # 将输入张量和分解参数进行乘法运算
    return tf.multiply(in_tensor, decompose_params)  # [batch_size, decompse_dim, dim]


def cal_maxpooling_matching(passage_rep, question_rep, decompose_params):
    # passage_representation: [batch_size, passage_len, dim]
    # qusetion_representation: [batch_size, question_len, dim]
    # decompose_params: [decompose_dim, dim]

    def singel_instance(x):
        p = x[0]
        q = x[1]
        # p: [pasasge_len, dim], q: [question_len, dim]
        p = multi_perspective_expand_for_2D(p, decompose_params)  # [pasasge_len, decompose_dim, dim]
        q = multi_perspective_expand_for_2D(q, decompose_params)  # [question_len, decompose_dim, dim]
        p = tf.expand_dims(p, 1)  # [pasasge_len, 1, decompose_dim, dim]
        q = tf.expand_dims(q, 0)  # [1, question_len, decompose_dim, dim]
        return cosine_distance(p, q)  # [passage_len, question_len, decompose]

    elems = (passage_rep, question_rep)
    matching_matrix = tf.map_fn(singel_instance, elems,
                                dtype=tf.float32)  # [batch_size, passage_len, question_len, decompse_dim]
    return tf.concat(axis=2, values=[tf.reduce_max(matching_matrix, axis=2), tf.reduce_mean(matching_matrix,
                                                                                            axis=2)])  # [batch_size, passage_len, 2*decompse_dim]


def cross_entropy(logits, truth, mask):
    # logits: [batch_size, passage_len]
    # truth: [batch_size, passage_len]
    # mask: [batch_size, passage_len]

    #     xdev = x - x.max()
    #     return xdev - T.log(T.sum(T.exp(xdev)))
    logits = tf.multiply(logits, mask)  # 将logits与mask相乘，得到新的logits
    xdev = tf.sub(logits, tf.expand_dims(tf.reduce_max(logits, 1), -1))  # 将logits减去每行的最大值，得到新的xdev
    log_predictions = tf.sub(xdev, tf.expand_dims(tf.log(tf.reduce_sum(tf.exp(xdev), -1)), -1))  # 将xdev减去每行的exp(xdev)的和的对数，得到新的log_predictions
    #     return -T.sum(targets * log_predictions)
    result = tf.multiply(tf.multiply(truth, log_predictions), mask)  # [batch_size, passage_len]
    return tf.multiply(-1.0, tf.reduce_sum(result, -1))  # [batch_size]


def highway_layer(in_val, output_size, scope=None):
    # in_val: [batch_size, passage_len, dim]
    input_shape = tf.shape(in_val)
    batch_size = input_shape[0]
    passage_len = input_shape[1]
    #     feat_dim = input_shape[2]
    in_val = tf.reshape(in_val, [batch_size * passage_len, output_size])
    with tf.variable_scope(scope or "highway_layer"):
        # 定义highway层的权重和偏置
        highway_w = tf.get_variable("highway_w", [output_size, output_size], dtype=tf.float32)
        # 定义highway层偏置
        highway_b = tf.get_variable("highway_b", [output_size], dtype=tf.float32)
        # 定义全连接层权重
        full_w = tf.get_variable("full_w", [output_size, output_size], dtype=tf.float32)
        # 定义全连接层偏置
        full_b = tf.get_variable("full_b", [output_size], dtype=tf.float32)
        # 计算tanh和sigmoid
        trans = tf.nn.tanh(tf.nn.xw_plus_b(in_val, full_w, full_b))
        gate = tf.nn.sigmoid(tf.nn.xw_plus_b(in_val, highway_w, highway_b))
        # 计算输出
        outputs = trans * gate + in_val * (1.0 - gate)
    outputs = tf.reshape(outputs, [batch_size, passage_len, output_size])
    return outputs


# 定义一个多层高速公路层函数，输入参数为in_val，输出大小output_size，层数num_layers，作用域scope
def multi_highway_layer(in_val, output_size, num_layers, scope=None):
    # 定义作用域名称
    scope_name = 'highway_layer'
    # 如果作用域不为空，则将作用域名称设置为scope
    if scope is not None: scope_name = scope
    # 循环num_layers次
    for i in range(num_layers):
        # 定义当前作用域名称
        cur_scope_name = scope_name + "-{}".format(i)
        # 调用高速公路层函数，将in_val作为输入，输出大小为output_size，作用域为cur_scope_name
        in_val = highway_layer(in_val, output_size, scope=cur_scope_name)
    # 返回in_val
    return in_val


def cal_max_question_representation(question_representation, atten_scores):
    # 计算atten_scores的最大值位置
    atten_positions = tf.argmax(atten_scores, axis=2, output_type=tf.int32)  # [batch_size, passage_len]
    # 根据atten_positions收集question_representation
    max_question_reps = layer_utils.collect_representation(question_representation, atten_positions)
    # 返回最大值位置的question_representation
    return max_question_reps


def multi_perspective_match(feature_dim, repres1, repres2, is_training=True, dropout_rate=0.2,
                            options=None, scope_name='mp-match', reuse=False):
    '''
        :param repres1: [batch_size, len, feature_dim]
        :param repres2: [batch_size, len, feature_dim]
        :return:
    '''
    # 获取输入形状
    input_shape = tf.shape(repres1)
    # 获取batch_size
    batch_size = input_shape[0]
    # 获取seq_length
    seq_length = input_shape[1]
    # 初始化匹配结果列表
    matching_result = []
    # 进入变量作用域
    with tf.variable_scope(scope_name, reuse=reuse):
        # 初始化匹配维度
        match_dim = 0
        # 如果使用余弦相似度
        if options['with_cosine']:
            # 计算余弦相似度
            cosine_value = layer_utils.cosine_distance(repres1, repres2, cosine_norm=False)
            # 将余弦相似度reshape为[batch_size, seq_length, 1]
            cosine_value = tf.reshape(cosine_value, [batch_size, seq_length, 1])
            # 将余弦相似度添加到匹配结果列表中
            matching_result.append(cosine_value)
            # 更新匹配维度
            match_dim += 1

        # 如果使用多通道余弦相似度
        if options['with_mp_cosine']:
            # 获取多通道余弦相似度参数
            mp_cosine_params = tf.get_variable("mp_cosine", shape=[options['cosine_MP_dim'], feature_dim],
                                               dtype=tf.float32)
            # 将多通道余弦相似度参数扩展为[1, 1, cosine_MP_dim, feature_dim]
            mp_cosine_params = tf.expand_dims(mp_cosine_params, axis=0)
            mp_cosine_params = tf.expand_dims(mp_cosine_params, axis=0)
            # 将repres1和repres2扩展为[batch_size, seq_length, 1, feature_dim]
            repres1_flat = tf.expand_dims(repres1, axis=2)
            repres2_flat = tf.expand_dims(repres2, axis=2)
            # 计算多通道余弦相似度
            mp_cosine_matching = layer_utils.cosine_distance(tf.multiply(repres1_flat, mp_cosine_params),
                                                             repres2_flat, cosine_norm=False)
            # 将多通道余弦相似度添加到匹配结果列表中
            matching_result.append(mp_cosine_matching)
            # 更新匹配维度
            match_dim += options['cosine_MP_dim']

    matching_result = tf.concat(axis=2, values=matching_result)
    return (matching_result, match_dim)


def match_passage_with_question(passage_reps, question_reps, passage_mask, question_mask, passage_lengths,
                                question_lengths,
                                context_lstm_dim, scope=None,
                                with_full_match=True, with_maxpool_match=True, with_attentive_match=True,
                                with_max_attentive_match=True,
                                is_training=True, options=None, dropout_rate=0, forward=True):
    # 将passage_reps和passage_mask相乘，得到passage_reps的mask
    passage_reps = tf.multiply(passage_reps, tf.expand_dims(passage_mask, -1))
    # 将question_reps和question_mask相乘，得到question_reps的mask
    question_reps = tf.multiply(question_reps, tf.expand_dims(question_mask, -1))
    # 创建一个空列表，用于存储所有的问题感知表示
    all_question_aware_representatins = []
    # 初始化dim为0
    dim = 0
    with tf.variable_scope(scope or "match_passage_with_question"):
        # 计算问题表示和段落表示的相关性矩阵
        relevancy_matrix = cal_relevancy_matrix(question_reps, passage_reps)
        # 对相关性矩阵进行掩码操作
        relevancy_matrix = mask_relevancy_matrix(relevancy_matrix, question_mask, passage_mask)
        # relevancy_matrix = layer_utils.calcuate_attention(passage_reps, question_reps, context_lstm_dim, context_lstm_dim,
        #             scope_name="fw_attention", att_type=options.att_type, att_dim=options.att_dim,
        #             remove_diagnoal=False, mask1=passage_mask, mask2=question_mask, is_training=is_training, dropout_rate=dropout_rate)

        # 将relevancy_matrix在axis=2维度上求最大值，并保持维度不变，将结果添加到all_question_aware_representatins列表中
        all_question_aware_representatins.append(tf.reduce_max(relevancy_matrix, axis=2, keep_dims=True))
        # 将relevancy_matrix在axis=2维度上求平均值，并保持维度不变，将结果添加到all_question_aware_representatins列表中
        all_question_aware_representatins.append(tf.reduce_mean(relevancy_matrix, axis=2, keep_dims=True))
        # 维度加2
        dim += 2
        if with_full_match:
            # 如果forward为True，则将question_reps的最后一个时间步的输出作为question_full_rep
            if forward:
                question_full_rep = layer_utils.collect_final_step_of_lstm(question_reps, question_lengths - 1)
            # 否则，将question_reps的第一个时间步的输出作为question_full_rep
            else:
                question_full_rep = question_reps[:, 0, :]

            # 获取passage_reps的长度
            passage_len = tf.shape(passage_reps)[1]
            # 将question_full_rep扩展为二维
            question_full_rep = tf.expand_dims(question_full_rep, axis=1)
            # 将question_full_rep复制passage_len次，使其与passage_reps的维度相同
            question_full_rep = tf.tile(question_full_rep,
                                        [1, passage_len, 1])  # [batch_size, pasasge_len, feature_dim]

            # 使用multi_perspective_match函数计算attentive_rep和match_dim
            (attentive_rep, match_dim) = multi_perspective_match(context_lstm_dim,
                                                                 passage_reps, question_full_rep,
                                                                 is_training=is_training,
                                                                 dropout_rate=options['dropout_rate'],
                                                                 options=options, scope_name='mp-match-full-match')
            # 将attentive_rep添加到all_question_aware_representatins列表中
            all_question_aware_representatins.append(attentive_rep)
            # 将match_dim加到dim中
            dim += match_dim

        # 如果使用最大池化匹配
        if with_maxpool_match:
            # 获取最大池化匹配的参数
            maxpooling_decomp_params = tf.get_variable("maxpooling_matching_decomp",
                                                       shape=[options['cosine_MP_dim'], context_lstm_dim],
                                                       dtype=tf.float32)
            # 计算最大池化匹配
            maxpooling_rep = cal_maxpooling_matching(passage_reps, question_reps, maxpooling_decomp_params)
            # 将最大池化匹配的结果添加到所有问题感知表示中
            all_question_aware_representatins.append(maxpooling_rep)
            # 更新维度
            dim += 2 * options['cosine_MP_dim']

        # 如果使用注意力匹配
        if with_attentive_match:
            # 计算注意力分数
            atten_scores = layer_utils.calcuate_attention(passage_reps, question_reps, context_lstm_dim,
                                                          context_lstm_dim,
                                                          scope_name="attention", att_type=options['att_type'],
                                                          att_dim=options['att_dim'],
                                                          remove_diagnoal=False, mask1=passage_mask,
                                                          mask2=question_mask, is_training=is_training,
                                                          dropout_rate=dropout_rate)
            # 计算注意力分数与问题表示的矩阵乘积
            att_question_contexts = tf.matmul(atten_scores, question_reps)
            # 计算多角度匹配
            (attentive_rep, match_dim) = multi_perspective_match(context_lstm_dim,
                                                                 passage_reps, att_question_contexts,
                                                                 is_training=is_training,
                                                                 dropout_rate=options['dropout_rate'],
                                                                 options=options, scope_name='mp-match-att_question')
            # 将多角度匹配的结果添加到所有问题感知表示中
            all_question_aware_representatins.append(attentive_rep)
            # 更新维度
            dim += match_dim

        # 如果使用最大注意力匹配
        if with_max_attentive_match:
            # 计算问题的最大表示
            max_att = cal_max_question_representation(question_reps, relevancy_matrix)
            # 计算最大注意力匹配
            (max_attentive_rep, match_dim) = multi_perspective_match(context_lstm_dim,
                                                                     passage_reps, max_att, is_training=is_training,
                                                                     dropout_rate=options['dropout_rate'],
                                                                     options=options, scope_name='mp-match-max-att')
            # 将最大注意力匹配的结果添加到所有问题的表示中
            all_question_aware_representatins.append(max_attentive_rep)
            # 更新维度
            dim += match_dim

        # 将所有问题感知表示进行拼接
        all_question_aware_representatins = tf.concat(axis=2, values=all_question_aware_representatins)
    return (all_question_aware_representatins, dim)


def bilateral_match_func(in_question_repres, in_passage_repres,
                         question_lengths, passage_lengths, question_mask, passage_mask, input_dim, is_training,
                         options=None):
    # 定义一个函数，用于进行双向匹配
    question_aware_representatins = []
    # 定义一个空列表，用于存储问题感知的表示
    question_aware_dim = 0
    # 定义一个变量，用于存储问题感知的维度
    passage_aware_representatins = []
    # 定义一个空列表，用于存储段落感知的表示
    passage_aware_dim = 0

    # ====word level matching======
    # 使用match_passage_with_question函数，将问题和段落进行匹配，得到匹配结果和匹配维度
    (match_reps, match_dim) = match_passage_with_question(in_passage_repres, in_question_repres, passage_mask,
                                                          question_mask, passage_lengths,
                                                          question_lengths, input_dim, scope="word_match_forward",
                                                          with_full_match=False,
                                                          with_maxpool_match=options['with_maxpool_match'],
                                                          with_attentive_match=options['with_attentive_match'],
                                                          with_max_attentive_match=options['with_max_attentive_match'],
                                                          is_training=is_training, options=options,
                                                          dropout_rate=options['dropout_rate'], forward=True)
    # 将匹配结果添加到问题感知表示中
    question_aware_representatins.append(match_reps)
    # 将匹配维度累加到问题感知维度中
    question_aware_dim += match_dim

    # 使用match_passage_with_question函数，将问题和段落进行匹配，得到匹配结果和匹配维度
    (match_reps, match_dim) = match_passage_with_question(in_question_repres, in_passage_repres, question_mask,
                                                          passage_mask, question_lengths,
                                                          passage_lengths, input_dim, scope="word_match_backward",
                                                          with_full_match=False,
                                                          with_maxpool_match=options['with_maxpool_match'],
                                                          with_attentive_match=options['with_attentive_match'],
                                                          with_max_attentive_match=options['with_max_attentive_match'],
                                                          is_training=is_training, options=options,
                                                          dropout_rate=options['dropout_rate'], forward=False)
    # 将匹配结果添加到段落感知表示中
    passage_aware_representatins.append(match_reps)
    # 将匹配维度累加到段落感知维度中
    passage_aware_dim += match_dim

    # 定义一个变量作用域，用于存储上下文匹配层
    with tf.variable_scope('context_MP_matching'):
        # 遍历上下文层的数量
        for i in range(options['context_layer_num']):  # support multiple context layer
            # 定义一个变量作用域，用于存储第i层的上下文匹配层
            with tf.variable_scope('layer-{}'.format(i)):
                # contextual lstm for both passage and question
                # 将in_question_repres和question_mask相乘，得到新的in_question_repres
                in_question_repres = tf.multiply(in_question_repres, tf.expand_dims(question_mask, axis=-1))
                # 将in_passage_repres和passage_mask相乘，得到新的in_passage_repres
                in_passage_repres = tf.multiply(in_passage_repres, tf.expand_dims(passage_mask, axis=-1))
                # 使用自定义的lstm层，对in_question_repres进行前向和后向的lstm计算，得到question_context_representation_fw、question_context_representation_bw和in_question_repres
                (question_context_representation_fw, question_context_representation_bw,
                 in_question_repres) = layer_utils.my_lstm_layer(
                    in_question_repres, options['context_lstm_dim'], input_lengths=question_lengths,
                    scope_name="context_represent",
                    reuse=False, is_training=is_training, dropout_rate=options['dropout_rate'],
                    use_cudnn=options['use_cudnn'])
                # 使用自定义的lstm层，对in_passage_repres进行前向和后向的lstm计算，得到passage_context_representation_fw、passage_context_representation_bw和in_passage_repres
                (passage_context_representation_fw, passage_context_representation_bw,
                 in_passage_repres) = layer_utils.my_lstm_layer(
                    in_passage_repres, options['context_lstm_dim'], input_lengths=passage_lengths,
                    scope_name="context_represent",
                    reuse=True, is_training=is_training, dropout_rate=options['dropout_rate'], use_cudnn=options['use_cudnn'])

                # Multi-perspective matching
                # 使用tf.variable_scope定义一个名为'left_MP_matching'的作用域
                with tf.variable_scope('left_MP_matching'):
                    # 调用match_passage_with_question函数，传入参数，得到match_reps和match_dim
                    (match_reps, match_dim) = match_passage_with_question(passage_context_representation_fw,
                                                                          question_context_representation_fw,
                                                                          passage_mask, question_mask, passage_lengths,
                                                                          question_lengths, options['context_lstm_dim'],
                                                                          scope="forward_match",
                                                                          with_full_match=options['with_full_match'],
                                                                          with_maxpool_match=options['with_maxpool_match'],
                                                                          with_attentive_match=options['with_attentive_match'],
                                                                          with_max_attentive_match=options['with_max_attentive_match'],
                                                                          is_training=is_training, options=options,
                                                                          dropout_rate=options['dropout_rate'],
                                                                          forward=True)
                    # 将match_reps添加到question_aware_representatins列表中
                    question_aware_representatins.append(match_reps)
                    # 将match_dim加到question_aware_dim中
                    question_aware_dim += match_dim
                    # 调用match_passage_with_question函数，传入参数，得到match_reps和match_dim
                    (match_reps, match_dim) = match_passage_with_question(passage_context_representation_bw,
                                                                          question_context_representation_bw,
                                                                          passage_mask, question_mask, passage_lengths,
                                                                          question_lengths, options['context_lstm_dim'],
                                                                          scope="backward_match",
                                                                          with_full_match=options['with_full_match'],
                                                                          with_maxpool_match=options['with_maxpool_match'],
                                                                          with_attentive_match=options['with_attentive_match'],
                                                                          with_max_attentive_match=options['with_max_attentive_match'],
                                                                          is_training=is_training, options=options,
                                                                          dropout_rate=options['dropout_rate'],
                                                                          forward=False)
                    # 将match_reps添加到question_aware_representatins列表中
                    question_aware_representatins.append(match_reps)
                    # 将match_dim加到question_aware_dim中
                    question_aware_dim += match_dim

                # 使用tf.variable_scope定义一个名为right_MP_matching的作用域
                with tf.variable_scope('right_MP_matching'):
                    # 调用match_passage_with_question函数，传入参数，计算question_context_representation_fw和passage_context_representation_fw的匹配结果
                    (match_reps, match_dim) = match_passage_with_question(question_context_representation_fw,
                                                                          passage_context_representation_fw,
                                                                          question_mask, passage_mask, question_lengths,
                                                                          passage_lengths, options['context_lstm_dim'],
                                                                          scope="forward_match",
                                                                          with_full_match=options['with_full_match'],
                                                                          with_maxpool_match=options['with_maxpool_match'],
                                                                          with_attentive_match=options['with_attentive_match'],
                                                                          with_max_attentive_match=options['with_max_attentive_match'],
                                                                          is_training=is_training, options=options,
                                                                          dropout_rate=options['dropout_rate'],
                                                                          forward=True)
                    # 将匹配结果添加到passage_aware_representatins列表中
                    passage_aware_representatins.append(match_reps)
                    # 将匹配结果的维度添加到passage_aware_dim中
                    passage_aware_dim += match_dim
                    # 调用match_passage_with_question函数，传入参数，计算question_context_representation_bw和passage_context_representation_bw的匹配结果
                    (match_reps, match_dim) = match_passage_with_question(question_context_representation_bw,
                                                                          passage_context_representation_bw,
                                                                          question_mask, passage_mask, question_lengths,
                                                                          passage_lengths, options['context_lstm_dim'],
                                                                          scope="backward_match",
                                                                          with_full_match=options['with_full_match'],
                                                                          with_maxpool_match=options['with_maxpool_match'],
                                                                          with_attentive_match=options['with_attentive_match'],
                                                                          with_max_attentive_match=options['with_max_attentive_match'],
                                                                          is_training=is_training, options=options,
                                                                          dropout_rate=options['dropout_rate'],
                                                                          forward=False)
                    # 将匹配结果添加到passage_aware_representatins列表中
                    passage_aware_representatins.append(match_reps)
                    # 将匹配结果的维度添加到passage_aware_dim中
                    passage_aware_dim += match_dim

    question_aware_representatins = tf.concat(axis=2,
                                              values=question_aware_representatins)  # [batch_size, passage_len, question_aware_dim]
    # 将question_aware_representatins在axis=2维度上进行拼接，得到新的question_aware_representatins
    passage_aware_representatins = tf.concat(axis=2,
                                             values=passage_aware_representatins)  # [batch_size, question_len, question_aware_dim]

    # 如果是在训练阶段，则对question_aware_representatins和passage_aware_representatins进行dropout操作
    if is_training:
        question_aware_representatins = tf.nn.dropout(question_aware_representatins, (1 - options['dropout_rate']))
        passage_aware_representatins = tf.nn.dropout(passage_aware_representatins, (1 - options['dropout_rate']))

    # ======Highway layer======
    # 如果使用match_highway，则对question_aware_representatins和passage_aware_representatins进行多highway层操作
    if options['with_match_highway']:
        with tf.variable_scope("left_matching_highway"):
            question_aware_representatins = multi_highway_layer(question_aware_representatins, question_aware_dim,
                                                                options['highway_layer_num'])
        with tf.variable_scope("right_matching_highway"):
            passage_aware_representatins = multi_highway_layer(passage_aware_representatins, passage_aware_dim,
                                                               options['highway_layer_num'])

    # ========Aggregation Layer======
    # 定义聚合表示列表
    aggregation_representation = []
    # 定义聚合维度
    aggregation_dim = 0

    # 定义问题感知聚合输入
    qa_aggregation_input = question_aware_representatins
    # 定义段落感知聚合输入
    pa_aggregation_input = passage_aware_representatins
    with tf.variable_scope('aggregation_layer'):
        # 遍历聚合层数量
        for i in range(options['aggregation_layer_num']):  # support multiple aggregation layer
            # 将qa_aggregation_input与passage_mask相乘
            qa_aggregation_input = tf.multiply(qa_aggregation_input, tf.expand_dims(passage_mask, axis=-1))
            # 使用自定义的LSTM层处理qa_aggregation_input
            (fw_rep, bw_rep, cur_aggregation_representation) = layer_utils.my_lstm_layer(
                qa_aggregation_input, options['aggregation_lstm_dim'], input_lengths=passage_lengths,
                scope_name='left_layer-{}'.format(i),
                reuse=False, is_training=is_training, dropout_rate=options['dropout_rate'], use_cudnn=options['use_cudnn'])
            # 获取LSTM层的最终输出
            fw_rep = layer_utils.collect_final_step_of_lstm(fw_rep, passage_lengths - 1)
            # 获取LSTM层的第一个输出
            bw_rep = bw_rep[:, 0, :]
            # 将LSTM层的输出添加到aggregation_representation中
            aggregation_representation.append(fw_rep)
            aggregation_representation.append(bw_rep)
            # 更新aggregation_dim
            aggregation_dim += 2 * options['aggregation_lstm_dim']
            # 更新qa_aggregation_input
            qa_aggregation_input = cur_aggregation_representation  # [batch_size, passage_len, 2*aggregation_lstm_dim]

            # 将pa_aggregation_input与question_mask相乘
            pa_aggregation_input = tf.multiply(pa_aggregation_input, tf.expand_dims(question_mask, axis=-1))
            # 使用自定义的LSTM层处理pa_aggregation_input
            (fw_rep, bw_rep, cur_aggregation_representation) = layer_utils.my_lstm_layer(
                pa_aggregation_input, options['aggregation_lstm_dim'],
                input_lengths=question_lengths, scope_name='right_layer-{}'.format(i),
                reuse=False, is_training=is_training, dropout_rate=options['dropout_rate'], use_cudnn=options['use_cudnn'])
            # 获取LSTM层的最终输出
            fw_rep = layer_utils.collect_final_step_of_lstm(fw_rep, question_lengths - 1)
            # 获取LSTM层的第一个输出
            bw_rep = bw_rep[:, 0, :]
            # 将LSTM层的输出添加到aggregation_representation中
            aggregation_representation.append(fw_rep)
            aggregation_representation.append(bw_rep)
            # 更新aggregation_dim
            aggregation_dim += 2 * options['aggregation_lstm_dim']
            # 更新pa_aggregation_input
            pa_aggregation_input = cur_aggregation_representation  # [batch_size, passage_len, 2*aggregation_lstm_dim]

    aggregation_representation = tf.concat(axis=1, values=aggregation_representation)  # [batch_size, aggregation_dim]

    # ======Highway layer======
    # 如果选项中包含聚合高速公路
    if options['with_aggregation_highway']:
        # 在变量作用域中创建聚合高速公路
        with tf.variable_scope("aggregation_highway"):
            # 获取聚合表示的形状
            agg_shape = tf.shape(aggregation_representation)
            # 获取批次大小
            batch_size = agg_shape[0]
            # 将聚合表示重新塑形
            aggregation_representation = tf.reshape(aggregation_representation, [1, batch_size, aggregation_dim])
            # 使用多高速公路层对聚合表示进行处理
            aggregation_representation = multi_highway_layer(aggregation_representation, aggregation_dim,
                                                             options['highway_layer_num'])
            # 将聚合表示重新塑形
            aggregation_representation = tf.reshape(aggregation_representation, [batch_size, aggregation_dim])

    # 返回聚合表示和聚合维度
    return (aggregation_representation, aggregation_dim)

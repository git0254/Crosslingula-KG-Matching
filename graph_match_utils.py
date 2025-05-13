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

def cal_relevancy_matrix(node_1_repres, node_2_repres, watch=None):
    # [batch_size, 1, single_graph_1_nodes_size, node_embedding_dim]
    node_1_repres_tmp = tf.expand_dims(node_1_repres, 1)

    # [batch_size, single_graph_2_nodes_size, 1, node_embedding_dim]
    node_2_repres_tmp = tf.expand_dims(node_2_repres, 2)

    # [batch_size, single_graph_2_nodes_size, single_graph_1_nodes_size]
    relevancy_matrix = cosine_distance(node_1_repres_tmp, node_2_repres_tmp)

    # 将node_1_repres_tmp、node_2_repres_tmp和relevancy_matrix存入watch字典中
    watch["node_1_repres_tmp"] = node_1_repres
    watch["node_2_repres_tmp"] = node_2_repres
    watch["relevancy_matrix"] = relevancy_matrix

    return relevancy_matrix

def mask_relevancy_matrix(relevancy_matrix, graph_1_mask, graph_2_mask):
    # relevancy_matrix: [batch_size, single_graph_2_nodes_size, single_graph_1_nodes_size]
    # graph_1_mask: [batch_size, single_graph_1_nodes_size]
    # graph_2_mask: [batch_size, single_graph_2_nodes_size]

    # 将relevancy_matrix与graph_1_mask相乘，得到新的relevancy_matrix
    relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(graph_1_mask, 1))
    # 将relevancy_matrix与graph_2_mask相乘，得到最终的relevancy_matrix
    relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(graph_2_mask, 2))

    # [batch_size, single_graph_2_nodes_size, single_graph_1_nodes_size]
    return relevancy_matrix

def multi_perspective_expand_for_2D(in_tensor, decompose_params):
    # [batch_size, 'x', dim]
    # 在in_tensor的axis=1处添加一个维度
    in_tensor = tf.expand_dims(in_tensor, axis=1)
    # [1, decompse_dim, dim]
    # 在decompose_params的axis=0处添加一个维度
    decompose_params = tf.expand_dims(decompose_params, axis=0)
    # [batch_size, decompse_dim, dim]
    return tf.multiply(in_tensor, decompose_params)

def cal_maxpooling_matching(node_1_rep, node_2_rep, decompose_params):
    # node_1_rep: [batch_size, single_graph_1_nodes_size, dim]
    # node_2_rep: [batch_size, single_graph_2_nodes_size, dim]
    # decompose_params: [decompose_dim, dim]
    def singel_instance(x):
        # p: [single_graph_1_nodes_size, dim], q: [single_graph_2_nodes_size, dim]
        p = x[0]
        q = x[1]

        # [single_graph_1_nodes_size, decompose_dim, dim]
        p = multi_perspective_expand_for_2D(p, decompose_params)

        # [single_graph_2_nodes_size, decompose_dim, dim]
        q = multi_perspective_expand_for_2D(q, decompose_params)

        # [single_graph_1_nodes_size, 1, decompose_dim, dim]
        p = tf.expand_dims(p, 1)

        # [1, single_graph_2_nodes_size, decompose_dim, dim]
        q = tf.expand_dims(q, 0)

        # [single_graph_1_nodes_size, single_graph_2_nodes_size, decompose]
        return cosine_distance(p, q)

    elems = (node_1_rep, node_2_rep)

    # [batch_size, single_graph_1_nodes_size, single_graph_2_nodes_size, decompse_dim]
    matching_matrix = tf.map_fn(singel_instance, elems, dtype=tf.float32)

    # [batch_size, single_graph_1_nodes_size, 2 * decompse_dim]
    return tf.concat(axis=2, values=[tf.reduce_max(matching_matrix, axis=2), tf.reduce_mean(matching_matrix, axis=2)])

def cal_max_node_2_representation(node_2_rep, relevancy_matrix):
    # [batch_size, single_graph_1_nodes_size]

    # 计算relevancy_matrix中每一行的最大值，并返回其索引
    atten_positions = tf.argmax(relevancy_matrix, axis=2, output_type=tf.int32)
    # 根据atten_positions从node_2_rep中收集对应的representation
    max_node_2_reps = layer_utils.collect_representation(node_2_rep, atten_positions)

    # [batch_size, single_graph_1_nodes_size, dim]
    # 返回max_node_2_reps
    return max_node_2_reps

def multi_perspective_match(feature_dim, rep_1, rep_2, options=None, scope_name='mp-match', reuse=False):
    '''
        :param repres1: [batch_size, len, feature_dim]
        :param repres2: [batch_size, len, feature_dim]
        :return:
    '''
    # 获取输入张量的形状
    input_shape = tf.shape(rep_1)
    # 获取批次大小
    batch_size = input_shape[0]
    # 获取序列长度
    seq_length = input_shape[1]
    # 初始化匹配结果列表
    matching_result = []
    # 创建变量作用域
    with tf.variable_scope(scope_name, reuse=reuse):
        match_dim = 0
        # 如果选项中包含cosine距离
        if options['with_cosine']:
            # 计算cosine距离
            cosine_value = layer_utils.cosine_distance(rep_1, rep_2, cosine_norm=False)
            # 将cosine距离reshape为[batch_size, seq_length, 1]
            cosine_value = tf.reshape(cosine_value, [batch_size, seq_length, 1])
            # 将cosine距离添加到匹配结果中
            matching_result.append(cosine_value)
            # 更新匹配维度
            match_dim += 1

        # 如果选项中包含多视角cosine距离
        if options['with_mp_cosine']:
            # 获取多视角cosine距离的参数
            mp_cosine_params = tf.get_variable("mp_cosine", shape=[options['cosine_MP_dim'], feature_dim],
                                               dtype=tf.float32)
            # 将参数扩展为[1, 1, cosine_MP_dim, feature_dim]
            mp_cosine_params = tf.expand_dims(mp_cosine_params, axis=0)
            mp_cosine_params = tf.expand_dims(mp_cosine_params, axis=0)
            # 将rep_1和rep_2扩展为[batch_size, seq_length, 1, feature_dim]
            rep_1_flat = tf.expand_dims(rep_1, axis=2)
            rep_2_flat = tf.expand_dims(rep_2, axis=2)
            # 计算多视角cosine距离
            mp_cosine_matching = layer_utils.cosine_distance(tf.multiply(rep_1_flat, mp_cosine_params),
                                                             rep_2_flat, cosine_norm=False)
            # 将多视角cosine距离添加到匹配结果中
            matching_result.append(mp_cosine_matching)
            # 更新匹配维度
            match_dim += options['cosine_MP_dim']

    # 将匹配结果拼接
    matching_result = tf.concat(axis=2, values=matching_result)
    # 返回匹配结果和匹配维度
    return (matching_result, match_dim)

def match_graph_1_with_graph_2(node_1_rep, node_2_rep, node_1_mask, node_2_mask, node_rep_dim, options=None, watch=None):
    '''

    :param node_1_rep:
    :param node_2_rep:
    :param node_1_mask:
    :param node_2_mask:
    :param node_rep_dim: dim of node representation
    :param with_maxpool_match:
    :param with_max_attentive_match:
    :param options:
    :return:
    '''

    # 从options中获取with_maxpool_match和with_max_attentive_match的值
    # 从options字典中获取with_maxpool_match的值
    with_maxpool_match = options["with_maxpool_match"]
    # 从options字典中获取with_max_attentive_match的值
    with_max_attentive_match = options["with_max_attentive_match"]

    # an array of [batch_size, single_graph_1_nodes_size]
    all_graph_2_aware_representations = []
    dim = 0
    with tf.variable_scope('match_graph_1_with_graph_2'):
        # [batch_size, single_graph_1_nodes_size, single_graph_2_nodes_size]
        # 计算节点2和节点1的关联矩阵
        relevancy_matrix = cal_relevancy_matrix(node_2_rep, node_1_rep, watch=watch)
        # 对关联矩阵进行掩码操作
        relevancy_matrix = mask_relevancy_matrix(relevancy_matrix, node_2_mask, node_1_mask)

        # 将关联矩阵的最大值和平均值添加到所有图2感知表示中
        all_graph_2_aware_representations.append(tf.reduce_max(relevancy_matrix, axis=2, keep_dims=True))
        all_graph_2_aware_representations.append(tf.reduce_mean(relevancy_matrix, axis=2, keep_dims=True))
        # 更新维度
        dim += 2

        # 如果使用最大池化匹配
        if with_maxpool_match:
            # 获取最大池化匹配的分解参数
            maxpooling_decomp_params = tf.get_variable("maxpooling_matching_decomp",
                                                       shape=[options['cosine_MP_dim'], node_rep_dim],
                                                       dtype=tf.float32)

            # [batch_size, single_graph_1_nodes_size, 2 * decompse_dim]
            # 计算最大池化匹配的表示
            maxpooling_rep = cal_maxpooling_matching(node_1_rep, node_2_rep, maxpooling_decomp_params)
            # 将最大池化匹配的表示与节点1的掩码相乘
            maxpooling_rep = tf.multiply(maxpooling_rep, tf.expand_dims(node_1_mask, -1))
            # 将最大池化匹配的表示添加到所有图2的表示中
            all_graph_2_aware_representations.append(maxpooling_rep)
            # 更新维度
            dim += 2 * options['cosine_MP_dim']

        if with_max_attentive_match:
            # [batch_size, single_graph_1_nodes_size, dim]
            # 计算节点2的表示
            max_att = cal_max_node_2_representation(node_2_rep, relevancy_matrix)

            # [batch_size, single_graph_1_nodes_size, match_dim]
            # 计算多角度匹配
            (max_attentive_rep, match_dim) = multi_perspective_match(node_rep_dim, node_1_rep, max_att, options=options, scope_name='mp-match-max-att')
            # 将节点1的掩码扩展到最后一维
            max_attentive_rep = tf.multiply(max_attentive_rep, tf.expand_dims(node_1_mask, -1))
            # 将匹配结果添加到列表中
            all_graph_2_aware_representations.append(max_attentive_rep)
            # 更新维度
            dim += match_dim

        # [batch_size, single_graph_1_nodes_size, dim]
        # 将所有图2的表示进行拼接
        all_graph_2_aware_representations = tf.concat(axis=2, values=all_graph_2_aware_representations)

    return (all_graph_2_aware_representations, dim)

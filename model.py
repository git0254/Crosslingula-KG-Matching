import tensorflow as tf
from neigh_samplers import UniformNeighborSampler
from aggregators import MeanAggregator, MaxPoolingAggregator, GatedMeanAggregator, GCNAggregator, SeqAggregator, AttentionAggregator
from graph_match_utils import match_graph_1_with_graph_2
from matching_model_options import options
from match_utils import multi_highway_layer
import numpy as np
import layer_utils

class GraphMatchNN(object):

    def __init__(self, mode, conf, pretrained_word_embeddings):
        #输入层的参数
        # 设置模型模式
        self.mode = mode
        # 设置词汇表大小
        self.word_vocab_size = conf.word_vocab_size
        # 设置L2正则化参数
        self.l2_lambda = conf.l2_lambda
        # 设置词嵌入维度
        self.word_embedding_dim = conf.hidden_layer_dim
        # 设置编码器隐藏层维度
        self.encoder_hidden_dim = conf.encoder_hidden_dim


        # the setting for the GCN
        # 设置层数
        self.num_layers = conf.num_layers
        # 设置图编码方向
        self.graph_encode_direction = conf.graph_encode_direction
        # 设置隐藏层维度
        self.hidden_layer_dim = conf.hidden_layer_dim
        # 设置是否拼接
        self.concat = conf.concat

        # 定义一个占位符，用于存储真实的标签
        self.y_true = tf.placeholder(tf.float32, [None, 2], name="true_labels")

        # the following place holders are for the first graph    第一张图（G1）的结构输入定义
        self.fw_adj_info_first = tf.placeholder(tf.int32, [None, None])               #正向邻接信息
        self.bw_adj_info_first = tf.placeholder(tf.int32, [None, None])               # 反向邻接信息
        self.feature_info_first = tf.placeholder(tf.int32, [None, None])              # 实体的词特征序列
        self.feature_len_first = tf.placeholder(tf.int32, [None])                     # 每个实体词序列的长度
        self.batch_nodes_first = tf.placeholder(tf.int32, [None, None])               # 当前batch中所有节点
        self.batch_mask_first = tf.placeholder(tf.float32, [None, None])              # 掩码信息
        self.looking_table_first = tf.placeholder(tf.int32, [None])                   # 实体索引映射
        self.entity_index_first = tf.placeholder(tf.int32, [None])                    # 每个图中目标实体的index
        #第二张图（G2）的结构输入定义
        self.fw_adj_info_second = tf.placeholder(tf.int32, [None, None])              # the fw adj info for each node
        self.bw_adj_info_second = tf.placeholder(tf.int32, [None, None])              # the bw adj info for each node
        self.feature_info_second = tf.placeholder(tf.int32, [None, None])             # the feature info for each node
        self.feature_len_second = tf.placeholder(tf.int32, [None])                    # the feature len for each node
        self.batch_nodes_second = tf.placeholder(tf.int32, [None, None])              # the nodes for the first batch
        self.batch_mask_second = tf.placeholder(tf.float32, [None, None])             # the mask for the second batch
        self.looking_table_second = tf.placeholder(tf.int32, [None])                  # the looking table for the second batch
        self.entity_index_second = tf.placeholder(tf.int32, [None])                   # the entity node index in each graph

        # 从配置文件中获取是否使用match highway
        self.with_match_highway = conf.with_match_highway
        # 从配置文件中获取是否使用gcn highway
        self.with_gcn_highway = conf.with_gcn_highway
        # 从配置文件中获取是否使用多个gcn 1状态
        self.if_use_multiple_gcn_1_state = conf.if_use_multiple_gcn_1_state
        # 从配置文件中获取是否使用多个gcn 2状态
        self.if_use_multiple_gcn_2_state = conf.if_use_multiple_gcn_2_state

        # 获取预训练的词嵌入
        self.pretrained_word_embeddings = pretrained_word_embeddings
        # 从配置文件中获取预训练词的大小
        self.pretrained_word_size = conf.pretrained_word_size
        # 从配置文件中获取学习词的大小
        self.learned_word_size = conf.learned_word_size

        # 获取fw_adj_info_first的样本大小
        self.sample_size_per_layer_first = tf.shape(self.fw_adj_info_first)[1]
        # 获取fw_adj_info_second的样本大小
        self.sample_size_per_layer_second = tf.shape(self.fw_adj_info_second)[1]
        # 获取y_true的样本大小
        self.batch_size = tf.shape(self.y_true)[0]
        # 从配置文件中获取dropout的值
        self.dropout = conf.dropout

        # 初始化前向聚合器列表
        self.fw_aggregators_first = []
        # 初始化反向聚合器列表
        self.bw_aggregators_first = []
        # 初始化聚合器维度
        self.aggregator_dim_first = conf.aggregator_dim_first
        # 初始化GCN窗口大小
        self.gcn_window_size_first = conf.gcn_window_size_first
        # 初始化GCN层大小
        self.gcn_layer_size_first = conf.gcn_layer_size_first

        # 初始化前向聚合器列表
        self.fw_aggregators_second = []
        # 初始化反向聚合器列表
        self.bw_aggregators_second = []
        # 初始化聚合器维度
        self.aggregator_dim_second = conf.aggregator_dim_second
        # 初始化GCN窗口大小
        self.gcn_window_size_second = conf.gcn_window_size_second
        # 初始化GCN层大小
        self.gcn_layer_size_second = conf.gcn_layer_size_second

        # 设置是否在开发集上进行预测
        self.if_pred_on_dev = False
        # 设置学习率
        self.learning_rate = conf.learning_rate

        # 设置相似度聚合方法
        self.agg_sim_method = conf.agg_sim_method

        # 设置GCN的第一层聚合类型
        self.agg_type_first = conf.gcn_type_first
        # 设置GCN的第二层聚合类型
        self.agg_type_second = conf.gcn_type_second

        # 设置余弦相似度矩阵的维度
        options['cosine_MP_dim'] = conf.cosine_MP_dim

        # 设置节点向量的方法
        self.node_vec_method = conf.node_vec_method
        # 设置预测方法
        self.pred_method = conf.pred_method
        # 设置观察对象
        self.watch = {}

    def _build_graph(self):
        """构建图神经网络计算图，包含节点编码、图卷积、图匹配、聚合和损失计算

               方法流程:
               1. 初始化节点掩码和查找表
               2. 构建词嵌入矩阵(包含PAD符号、预训练嵌入、可训练嵌入)
               3. 通过LSTM或词嵌入方式编码节点特征
               4. 使用图卷积网络(GCN)聚合邻居信息
               5. 应用掩码过滤无效节点
               6. 根据预测方法选择节点级别或图级别预测分支
               7. 图级别分支包含跨图匹配、特征聚合、多层感知等操作
               8. 构建损失函数和训练操作
               """
        # ======================初始化节点掩码和查找表============================================
        node_1_mask = self.batch_mask_first #图1的批量掩码（batch_size， node_num）
        node_2_mask = self.batch_mask_second    ## 图2节点的批量掩码
        node_1_looking_table = self.looking_table_first # 图1节点的邻接查找表
        node_2_looking_table = self.looking_table_second    # 图2节点的邻接查找表

        node_2_aware_representations = []
        node_2_aware_dim = 0
        node_1_aware_representations = []
        node_1_aware_dim = 0
        # ======================== 构建词嵌入矩阵============================================
        # 合并PAD符号嵌入、预训练词嵌入、可训练词嵌入三部分
        pad_word_embedding = tf.zeros([1, self.word_embedding_dim])  # PAD符号的零嵌入
        #   最终词嵌入矩阵形状：[total_vocab_size, embedding_dim]
        self.word_embeddings = tf.concat([pad_word_embedding,
                                          tf.get_variable('pretrained_embedding', shape=[self.pretrained_word_size, self.word_embedding_dim],
                                                          initializer=tf.constant_initializer(self.pretrained_word_embeddings), trainable=True),
                                          tf.get_variable('W_train',
                                                          shape=[self.learned_word_size, self.word_embedding_dim],
                                                          initializer=tf.contrib.layers.xavier_initializer(),
                                                          trainable=True)], 0)

        self.watch['word_embeddings'] = self.word_embeddings

        # ============ encode node feature by looking up word embedding =============
        # ================== 节点特征编码 ==================
        with tf.variable_scope('node_rep_gen'):
            # [node_size, hidden_layer_dim]
            feature_embedded_chars_first = tf.nn.embedding_lookup(self.word_embeddings, self.feature_info_first)
            graph_1_size = tf.shape(feature_embedded_chars_first)[0]

            feature_embedded_chars_second = tf.nn.embedding_lookup(self.word_embeddings, self.feature_info_second)
            graph_2_size = tf.shape(feature_embedded_chars_second)[0]

            #使用LSTM编码特征序列，取最后时间步作为节点表示
            if self.node_vec_method == "lstm":
                # 构建LSTM编码器单元
                cell = self.build_encoder_cell(1, self.hidden_layer_dim)

                # 对第一个图的特征嵌入进行动态RNN计算
                outputs, hidden_states = tf.nn.dynamic_rnn(cell=cell, inputs=feature_embedded_chars_first,
                                                           sequence_length=self.feature_len_first, dtype=tf.float32)
                # 收集LSTM输出的最后一个时间步作为节点表示
                node_1_rep = layer_utils.collect_final_step_of_lstm(outputs, self.feature_len_first-1)

                # 对第二个图的特征嵌入进行动态RNN计算
                outputs, hidden_states = tf.nn.dynamic_rnn(cell=cell, inputs=feature_embedded_chars_second,
                                                           sequence_length=self.feature_len_second, dtype=tf.float32)
                # 收集LSTM输出的最后一个时间步作为节点表示
                node_2_rep = layer_utils.collect_final_step_of_lstm(outputs, self.feature_len_second-1)

            elif self.node_vec_method == "word_emb":
                # 直接展平词嵌入作为节点表示
                node_1_rep = tf.reshape(feature_embedded_chars_first, [graph_1_size, -1])
                node_2_rep = tf.reshape(feature_embedded_chars_second, [graph_2_size, -1])

            # 将第一个图的节点表示保存到watch字典中
            self.watch["node_1_rep_initial"] = node_1_rep

        # ============ encode node feature by GCN =============
        # ============ 图卷积编码 =============
        with tf.variable_scope('first_gcn') as first_gcn_scope:
            # shape of node embedding: [batch_size, single_graph_nodes_size, node_embedding_dim]
            # shape of node size: [batch_size]
            # 对两个图分别进行GCN编码，聚合邻居信息
            gcn_1_res = self.gcn_encode(self.batch_nodes_first,         # 包含节点表示、最大池化、平均池化等输出
                                        node_1_rep,
                                        self.fw_adj_info_first, self.bw_adj_info_first,
                                        input_node_dim=self.word_embedding_dim,
                                        output_node_dim=self.aggregator_dim_first,
                                        fw_aggregators=self.fw_aggregators_first,
                                        bw_aggregators=self.bw_aggregators_first,
                                        window_size=self.gcn_window_size_first,
                                        layer_size=self.gcn_layer_size_first,
                                        scope="first_gcn",
                                        agg_type=self.agg_type_first,
                                        sample_size_per_layer=self.sample_size_per_layer_first,
                                        keep_inter_state=self.if_use_multiple_gcn_1_state)

            node_1_rep = gcn_1_res[0]
            node_1_rep_dim = gcn_1_res[3]

            gcn_2_res = self.gcn_encode(self.batch_nodes_second,
                                        node_2_rep,
                                        self.fw_adj_info_second,
                                        self.bw_adj_info_second,
                                        input_node_dim=self.word_embedding_dim,
                                        output_node_dim=self.aggregator_dim_first,
                                        fw_aggregators=self.fw_aggregators_first,
                                        bw_aggregators=self.bw_aggregators_first,
                                        window_size=self.gcn_window_size_first,
                                        layer_size=self.gcn_layer_size_first,
                                        scope="first_gcn",
                                        agg_type=self.agg_type_first,
                                        sample_size_per_layer=self.sample_size_per_layer_second,
                                        keep_inter_state=self.if_use_multiple_gcn_1_state)

            node_2_rep = gcn_2_res[0]
            node_2_rep_dim = gcn_2_res[3]

        self.watch["node_1_rep_first_GCN"] = node_1_rep
        self.watch["node_1_mask"] = node_1_mask

        # mask
        #========================掩码应用========================
        # 过滤无效节点(PAD节点)，保留有效节点表示
        node_1_rep = tf.multiply(node_1_rep, tf.expand_dims(node_1_mask, 2))
        node_2_rep = tf.multiply(node_2_rep, tf.expand_dims(node_2_mask, 2))

        self.watch["node_1_rep_first_GCN_masked"] = node_1_rep

        # ========================预测分支选择========================
        if self.pred_method == "node_level":
            # 节点级别预测：直接比较两个图的实体表示
            # 提取图1的实体表示
            entity_1_rep = tf.reshape(tf.nn.embedding_lookup(tf.transpose(node_1_rep, [1, 0, 2]), tf.constant(0)), [-1, node_1_rep_dim])
            # 提取图2的实体表示
            entity_2_rep = tf.reshape(tf.nn.embedding_lookup(tf.transpose(node_2_rep, [1, 0, 2]), tf.constant(0)), [-1, node_2_rep_dim])

            entity_1_2_diff = entity_1_rep - entity_2_rep
            entity_1_2_sim = entity_1_rep * entity_2_rep

            aggregation = tf.concat([entity_1_rep, entity_2_rep, entity_1_2_diff, entity_1_2_sim], axis=1)
            aggregation_dim = 4 * node_1_rep_dim

            w_0 = tf.get_variable("w_0", [aggregation_dim, aggregation_dim / 2], dtype=tf.float32)
            b_0 = tf.get_variable("b_0", [aggregation_dim / 2], dtype=tf.float32)
            w_1 = tf.get_variable("w_1", [aggregation_dim / 2, 2], dtype=tf.float32)
            b_1 = tf.get_variable("b_1", [2], dtype=tf.float32)

            # ====== Prediction Layer ===============
            logits = tf.matmul(aggregation, w_0) + b_0
            logits = tf.tanh(logits)
            logits = tf.matmul(logits, w_1) + b_1

        elif self.pred_method == "graph_level":
            # if the prediction method is graph_level, we perform the graph matching based prediction
            # 图级别预测：进行跨图匹配和多层聚合

            assert node_1_rep_dim == node_2_rep_dim
            input_dim = node_1_rep_dim

            with tf.variable_scope('node_level_matching') as matching_scope:
                # ========= node level matching ===============
                # 双向图匹配：图1节点与图2上下文匹配，图2节点与图1上下文匹配
                # match_graph_1_with_graph_2函数用于将图1的节点与图2的上下文进行匹配，返回匹配后的表示和维度
                (match_reps, match_dim) = match_graph_1_with_graph_2(node_1_rep, node_2_rep, node_1_mask, node_2_mask, input_dim,
                                                                     options=options, watch=self.watch)

                # 重用变量
                matching_scope.reuse_variables()

                # 将匹配后的表示添加到node_2_aware_representations列表中
                node_2_aware_representations.append(match_reps)
                # 将匹配后的维度添加到node_2_aware_dim中
                node_2_aware_dim += match_dim

                # 将图2的节点与图1的上下文进行匹配
                (match_reps, match_dim) = match_graph_1_with_graph_2(node_2_rep, node_1_rep, node_2_mask, node_1_mask, input_dim,
                                                                     options=options, watch=self.watch)

                # 将匹配后的表示添加到node_1_aware_representations列表中
                node_1_aware_representations.append(match_reps)
                # 将匹配后的维度添加到node_1_aware_dim中
                node_1_aware_dim += match_dim

            # TODO: add one more MP matching over the graph representation
            # with tf.variable_scope('context_MP_matching'):
            #     for i in range(options['context_layer_num']):
            #         with tf.variable_scope('layer-{}',format(i)):

            # [batch_size, single_graph_nodes_size, node_2_aware_dim]
            # ========= 多层级特征聚合 ========
            # 拼接不同匹配层结果，应用Highway网络增强特征
            node_2_aware_representations = tf.concat(axis=2, values=node_2_aware_representations)   #图1节点对图2的感知表示

            # [batch_size, single_graph_nodes_size, node_1_aware_dim]
            node_1_aware_representations = tf.concat(axis=2, values=node_1_aware_representations)

            # if self.mode == "train":
            #     node_2_aware_representations = tf.nn.dropout(node_2_aware_representations, (1 - options['dropout_rate']))
            #     node_1_aware_representations = tf.nn.dropout(node_1_aware_representations, (1 - options['dropout_rate']))

            # ========= Highway layer ==============
            # 如果使用匹配高速公路
            if self.with_match_highway:
                # 在变量作用域中创建左匹配高速公路
                with tf.variable_scope("left_matching_highway"):
                    # 使用多高速公路层对节点2感知表示进行转换
                    node_2_aware_representations = multi_highway_layer(node_2_aware_representations, node_2_aware_dim,
                                                                        options['highway_layer_num'])
                # 在变量作用域中创建右匹配高速公路
                with tf.variable_scope("right_matching_highway"):
                    # 使用多高速公路层对节点1感知表示进行转换
                    node_1_aware_representations = multi_highway_layer(node_1_aware_representations, node_1_aware_dim,
                                                                       options['highway_layer_num'])

            # 将节点2感知表示存储在watch字典中
            self.watch["node_1_rep_match"] = node_2_aware_representations

            # ========= Aggregation Layer ==============
            # 初始化聚合表示和聚合维度
            aggregation_representation = []
            aggregation_dim = 0

            # 获取节点2感知的聚合输入
            node_2_aware_aggregation_input = node_2_aware_representations
            # 获取节点1感知的聚合输入
            node_1_aware_aggregation_input = node_1_aware_representations

            # 监控节点1感知匹配层的输入
            self.watch["node_1_rep_match_layer"] = node_2_aware_aggregation_input

            # ========= GCN二次聚合 ========
            with tf.variable_scope('aggregation_layer'):
                # TODO: now we only have 1 aggregation layer; need to change this part if support more aggregation layers
                # [batch_size, single_graph_nodes_size, node_2_aware_dim]
                node_2_aware_aggregation_input = tf.multiply(node_2_aware_aggregation_input,
                                                             tf.expand_dims(node_1_mask, axis=-1))

                # [batch_size, single_graph_nodes_size, node_1_aware_dim]
                node_1_aware_aggregation_input = tf.multiply(node_1_aware_aggregation_input,
                                                             tf.expand_dims(node_2_mask, axis=-1))

                if self.agg_sim_method == "GCN":
                    # [batch_size*single_graph_nodes_size, node_2_aware_dim]
                    node_2_aware_aggregation_input = tf.reshape(node_2_aware_aggregation_input,
                                                                shape=[-1, node_2_aware_dim])

                    # [batch_size*single_graph_nodes_size, node_1_aware_dim]
                    node_1_aware_aggregation_input = tf.reshape(node_1_aware_aggregation_input,
                                                                shape=[-1, node_1_aware_dim])

                    # [node_1_size, node_2_aware_dim]
                    node_1_rep = tf.concat([tf.nn.embedding_lookup(node_2_aware_aggregation_input, node_1_looking_table),
                                            tf.zeros([1, node_2_aware_dim])], 0)

                    # [node_2_size, node_1_aware_dim]
                    node_2_rep = tf.concat([tf.nn.embedding_lookup(node_1_aware_aggregation_input, node_2_looking_table),
                                            tf.zeros([1, node_1_aware_dim])], 0)

                    # 使用GCN编码
                    gcn_1_res = self.gcn_encode(self.batch_nodes_first,
                                                node_1_rep,
                                                self.fw_adj_info_first,
                                                self.bw_adj_info_first,
                                                input_node_dim=node_2_aware_dim,
                                                output_node_dim=self.aggregator_dim_second,
                                                fw_aggregators=self.fw_aggregators_second,
                                                bw_aggregators=self.bw_aggregators_second,
                                                window_size=self.gcn_window_size_second,
                                                layer_size=self.gcn_layer_size_second,
                                                scope="second_gcn",
                                                agg_type=self.agg_type_second,
                                                sample_size_per_layer=self.sample_size_per_layer_first,
                                                keep_inter_state=self.if_use_multiple_gcn_2_state)

                    # 获取GCN_1的结果，其中gcn_1_res[1]为最大图表示，gcn_1_res[2]为平均图表示，gcn_1_res[3]为图表示的维度
                    max_graph_1_rep = gcn_1_res[1]
                    mean_graph_1_rep = gcn_1_res[2]
                    graph_1_rep_dim = gcn_1_res[3]

                    # 使用GCN编码器对第二层节点进行编码，得到第二层节点的表示
                    gcn_2_res = self.gcn_encode(self.batch_nodes_second,
                                                node_2_rep,
                                                self.fw_adj_info_second,
                                                self.bw_adj_info_second,
                                                input_node_dim=node_1_aware_dim,
                                                output_node_dim=self.aggregator_dim_second,
                                                fw_aggregators=self.fw_aggregators_second,
                                                bw_aggregators=self.bw_aggregators_second,
                                                window_size=self.gcn_window_size_second,
                                                layer_size=self.gcn_layer_size_second,
                                                scope="second_gcn",
                                                agg_type=self.agg_type_second,
                                                sample_size_per_layer=self.sample_size_per_layer_second,
                                                keep_inter_state=self.if_use_multiple_gcn_2_state)

                    # 获取第二层节点的最大表示
                    max_graph_2_rep = gcn_2_res[1]
                    # 获取第二层节点的平均表示
                    mean_graph_2_rep = gcn_2_res[2]
                    # 获取第二层节点的表示维度
                    graph_2_rep_dim = gcn_2_res[3]

                    # 断言 graph_1_rep_dim 等于 graph_2_rep_dim
                    assert graph_1_rep_dim == graph_2_rep_dim

                    # 如果使用多个GCN状态
                    if self.if_use_multiple_gcn_2_state:
                        # 获取第一个GCN的表示
                        graph_1_reps = gcn_1_res[5]
                        # 获取第二个GCN的表示
                        graph_2_reps = gcn_2_res[5]
                        # 获取交互维度
                        inter_dims = gcn_1_res[6]
                        # 遍历第一个GCN的表示
                        for idx in range(len(graph_1_reps)):
                            # 获取第一个GCN的最大表示和平均表示
                            (max_graph_1_rep_tmp, mean_graph_1_rep_tmp) = graph_1_reps[idx]
                            # 获取第二个GCN的最大表示和平均表示
                            (max_graph_2_rep_tmp, mean_graph_2_rep_tmp) = graph_2_reps[idx]
                            # 获取交互维度
                            inter_dim = inter_dims[idx]
                            # 将第一个GCN的最大表示和平均表示添加到聚合表示中
                            aggregation_representation.append(max_graph_1_rep_tmp)
                            aggregation_representation.append(mean_graph_1_rep_tmp)
                            # 将第二个GCN的最大表示和平均表示添加到聚合表示中
                            aggregation_representation.append(max_graph_2_rep_tmp)
                            aggregation_representation.append(mean_graph_2_rep_tmp)
                            # 更新聚合维度
                            aggregation_dim += 4 * inter_dim

                    # 如果不使用多个GCN状态
                    else:
                        # 将第一个GCN的最大表示和平均表示添加到聚合表示中
                        aggregation_representation.append(max_graph_1_rep)
                        aggregation_representation.append(mean_graph_1_rep)
                        # 将第二个GCN的最大表示和平均表示添加到聚合表示中
                        aggregation_representation.append(max_graph_2_rep)
                        aggregation_representation.append(mean_graph_2_rep)
                        # 更新聚合维度
                        aggregation_dim = 4 * graph_1_rep_dim

                    # aggregation_representation = tf.concat(aggregation_representation, axis=1)

                    # 计算gcn_2_window_size的值，即aggregation_representation的长度除以4
                    gcn_2_window_size = int(len(aggregation_representation)/4)
                    # 计算aggregation_dim的值，即aggregation_dim除以gcn_2_window_size
                    aggregation_dim = aggregation_dim/gcn_2_window_size

                    # 定义w_0变量，形状为[aggregation_dim, aggregation_dim / 2]，数据类型为tf.float32
                    w_0 = tf.get_variable("w_0", [aggregation_dim, aggregation_dim / 2], dtype=tf.float32)
                    # 定义b_0变量，形状为[aggregation_dim / 2]，数据类型为tf.float32
                    b_0 = tf.get_variable("b_0", [aggregation_dim / 2], dtype=tf.float32)
                    # 定义w_1变量，形状为[aggregation_dim / 2, 2]，数据类型为tf.float32
                    w_1 = tf.get_variable("w_1", [aggregation_dim / 2, 2], dtype=tf.float32)
                    # 定义b_1变量，形状为[2]，数据类型为tf.float32
                    b_1 = tf.get_variable("b_1", [2], dtype=tf.float32)

                    # 定义weights变量，形状为[gcn_2_window_size]，数据类型为tf.float32
                    weights = tf.get_variable("gcn_2_window_weights", [gcn_2_window_size], dtype=tf.float32)

                    # shape: [gcn_2_window_size, batch_size, 2]
                    logits = []
                    for layer_idx in range(gcn_2_window_size):
                        # 获取每个层的聚合表示
                        max_graph_1_rep = aggregation_representation[layer_idx * 4 + 0]
                        mean_graph_1_rep = aggregation_representation[layer_idx * 4 + 1]
                        max_graph_2_rep = aggregation_representation[layer_idx * 4 + 2]
                        mean_graph_2_rep = aggregation_representation[layer_idx * 4 + 3]

                        # 将每个层的聚合表示进行拼接
                        aggregation_representation_single = tf.concat([max_graph_1_rep, mean_graph_1_rep, max_graph_2_rep, mean_graph_2_rep], axis=1)

                        # ====== Prediction Layer ===============
                        logit = tf.matmul(aggregation_representation_single, w_0) + b_0
                        logit = tf.tanh(logit)
                        logit = tf.matmul(logit, w_1) + b_1
                        logits.append(logit)

                    # 如果logits的长度不等于1，则进行以下操作
                    if len(logits) != 1:
                        # 将logits进行拼接，并重新调整形状
                        logits = tf.reshape(tf.concat(logits, axis=0), [gcn_2_window_size, -1, 2])
                        # 调整logits的维度顺序
                        logits = tf.transpose(logits, [1, 0, 2])
                        # 将logits与weights相乘
                        logits = tf.multiply(logits, tf.expand_dims(weights, axis=-1))
                        # 沿着第一个维度求和
                        logits = tf.reduce_sum(logits, axis=1)
                    # 如果logits的长度等于1，则直接调整logits的形状
                    else:
                        logits = tf.reshape(logits, [-1, 2])



        # ====== Highway layer ============
        # if options['with_aggregation_highway']:

        # 定义损失函数
        with tf.name_scope("loss"):
            # 计算预测值
            self.y_pred = tf.nn.softmax(logits)
            # 计算交叉熵损失
            self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_true, logits=logits, name="xentropy_loss")) / tf.cast(self.batch_size, tf.float32)

        # ============  Training Objective ===========================
        # 如果当前模式为训练，并且不进行开发集预测
        if self.mode == "train" and not self.if_pred_on_dev:
            # 使用Adam优化器
            optimizer = tf.train.AdamOptimizer()
            # 获取所有可训练的变量
            params = tf.trainable_variables()
            # 计算损失函数对每个变量的梯度
            gradients = tf.gradients(self.loss, params)
            # 对梯度进行裁剪，防止梯度爆炸
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1)
            # 使用优化器更新参数
            self.training_op = optimizer.apply_gradients(zip(clipped_gradients, params))

    def build_encoder_cell(self, num_layers, hidden_size):
        """构建编码器RNN单元

                根据指定的层数构建单层或多层LSTM单元，训练模式下自动添加dropout

                Args:
                    num_layers (int): RNN的堆叠层数，必须为正整数
                    hidden_size (int): LSTM单元隐藏层维度大小

                Returns:
                    tf.nn.rnn_cell.RNNCell: 单层LSTM单元或多层堆叠的MultiRNNCell
                """
        # 处理单层LSTM情况
        if num_layers == 1:
            # 创建基础LSTM单元
            cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
            # 训练模式且dropout率有效时添加dropout层
            if self.mode == "train" and self.dropout > 0.0:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, self.dropout)
            return cell
        # 处理多层LSTM堆叠情况
        else:
            cell_list = []
            # 逐层创建LSTM单元并添加到列表中
            for i in range(num_layers):
                single_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
                # 每个层单独配置dropout
                if self.mode == "train" and self.dropout > 0.0:
                    single_cell = tf.nn.rnn_cell.DropoutWrapper(single_cell, self.dropout)
                cell_list.append(single_cell)
                # 将多个LSTM单元堆叠为多层结构
            return tf.contrib.rnn.MultiRNNCell(cell_list)

    def gcn_encode(self, batch_nodes, embedded_node_rep, fw_adj_info, bw_adj_info, input_node_dim, output_node_dim, fw_aggregators, bw_aggregators, window_size, layer_size, scope, agg_type, sample_size_per_layer, keep_inter_state=False):
        """
               图卷积网络编码函数，实现多层的图结构信息聚合

               参数:
               batch_nodes: Tensor, 当前批次的节点ID集合 [batch_size, node_per_graph]
               embedded_node_rep: Tensor, 节点初始嵌入表示矩阵 [node_num, embedding_dim]
               fw_adj_info/bw_adj_info: 前向/后向图的邻接关系信息
               input_node_dim: int, 输入节点特征维度
               output_node_dim: int, 输出节点特征维度
               fw_aggregators/bw_aggregators: list, 前向/后向各层的聚合器实例列表
               window_size: int, 滑动窗口大小控制信息传播步数
               layer_size: int, 图卷积层数
               scope: str, 变量作用域名称
               agg_type: str, 聚合器类型（"GCN"/"mean_pooling"/"max_pooling"/"lstm"/"att"）
               sample_size_per_layer: int, 每层采样的邻居节点数量
               keep_inter_state: bool, 是否保留中间层状态

               返回值:
               list: 包含以下元素的列表：
                   - hidden: 最终节点表示 [batch_size, node_num, graph_dim]
                   - max_graph_embedding: 最大池化图表示 [batch_size, graph_dim]
                   - mean_graph_embedding: 平均池化图表示 [batch_size, graph_dim]
                   - graph_dim: 最终图表示维度
                   - inter_node_reps: 中间层节点表示列表（仅当keep_inter_state=True时存在）
                   - inter_graph_reps: 中间层图表示列表（仅当keep_inter_state=True时存在）
                   - inter_graph_dims: 中间层维度列表（仅当keep_inter_state=True时存在）
               """
        with tf.variable_scope(scope):
            # 初始化基础参数
            single_graph_nodes_size = tf.shape(batch_nodes)[1]
            # ============ 图结构编码核心流程 ==========
            # 初始化邻居采样器
            fw_sampler = UniformNeighborSampler(fw_adj_info)
            bw_sampler = UniformNeighborSampler(bw_adj_info)
            # 展平批次维度用于嵌入查找
            nodes = tf.reshape(batch_nodes, [-1, ])

            # the fw_hidden and bw_hidden is the initial node embedding
            # [node_size, dim_size]
            # 初始化节点表示
            # 根据节点索引，从嵌入节点表示中查找对应的嵌入向量
            fw_hidden = tf.nn.embedding_lookup(embedded_node_rep, nodes)
            bw_hidden = tf.nn.embedding_lookup(embedded_node_rep, nodes)

            # [node_size, adj_size]
            # 采样邻居节点
            fw_sampled_neighbors = fw_sampler((nodes, sample_size_per_layer))
            bw_sampled_neighbors = bw_sampler((nodes, sample_size_per_layer))

            # 中间状态存储
            inter_fw_hiddens = []
            inter_bw_hiddens = []
            inter_dims = []

            if scope == "first_gcn":
                self.watch["node_1_rep_in_first_gcn"] = []

            # 多层图卷积处理
            fw_hidden_dim = input_node_dim
            # layer is the index of convolution and hop is used to combine information
            for layer in range(layer_size):
                self.watch["node_1_rep_in_first_gcn"].append(fw_hidden)
                # 按需创建聚合器实例
                if len(fw_aggregators) <= layer:
                    fw_aggregators.append([])
                if len(bw_aggregators) <= layer:
                    bw_aggregators.append([])
                # 窗口滑动聚合过程
                for hop in range(window_size):
                    # 如果hop大于6，则fw_aggregator为fw_aggregators[layer][6]
                    if hop > 6:
                        fw_aggregator = fw_aggregators[layer][6]
                    # 如果fw_aggregators[layer]的长度大于hop，则fw_aggregator为fw_aggregators[layer][hop]
                    elif len(fw_aggregators[layer]) > hop:
                        fw_aggregator = fw_aggregators[layer][hop]
                    # 否则，根据agg_type的值，选择不同的aggregator
                    else:
                        # 如果agg_type为"GCN"，则fw_aggregator为GCNAggregator
                        if agg_type == "GCN":
                            fw_aggregator = GCNAggregator(fw_hidden_dim, output_node_dim, concat=self.concat,
                                                          dropout=self.dropout, mode=self.mode)
                        # 如果agg_type为"mean_pooling"，则fw_aggregator为MeanAggregator
                        elif agg_type == "mean_pooling":
                            fw_aggregator = MeanAggregator(fw_hidden_dim, output_node_dim, concat=self.concat,
                                                           dropout=self.dropout, if_use_high_way=self.with_gcn_highway, mode=self.mode)
                        # 如果agg_type为"max_pooling"，则fw_aggregator为MaxPoolingAggregator
                        elif agg_type == "max_pooling":
                            fw_aggregator = MaxPoolingAggregator(fw_hidden_dim, output_node_dim, concat=self.concat,
                                                                 dropout=self.dropout, mode=self.mode)
                        # 如果agg_type为"lstm"，则fw_aggregator为SeqAggregator
                        elif agg_type == "lstm":
                            fw_aggregator = SeqAggregator(fw_hidden_dim, output_node_dim, concat=self.concat,
                                                          dropout=self.dropout, mode=self.mode)
                        # 如果agg_type为"att"，则fw_aggregator为AttentionAggregator
                        elif agg_type == "att":
                            fw_aggregator = AttentionAggregator(fw_hidden_dim, output_node_dim, concat=self.concat,
                                                                dropout=self.dropout, mode=self.mode)

                        fw_aggregators[layer].append(fw_aggregator)

                    # [node_size, adj_size, word_embedding_dim]
                    # 前向传播聚合
                    # 如果当前层为0且跳数为0，则从嵌入节点表示中查找前向采样邻居
                    if layer == 0 and hop == 0:
                        neigh_vec_hidden = tf.nn.embedding_lookup(embedded_node_rep, fw_sampled_neighbors)
                    # 否则，将前向隐藏状态和全零向量拼接，并从拼接结果中查找前向采样邻居
                    else:
                        neigh_vec_hidden = tf.nn.embedding_lookup(
                            tf.concat([fw_hidden, tf.zeros([1, fw_hidden_dim])], 0), fw_sampled_neighbors)

                    # if self.with_gcn_highway:
                    #     # we try to forget something when introducing the neighbor information
                    #     with tf.variable_scope("fw_hidden_highway"):
                    #         fw_hidden = multi_highway_layer(fw_hidden, fw_hidden_dim, options['highway_layer_num'])

                    # 设置前向隐藏层维度等于前向隐藏层维度
                    bw_hidden_dim = fw_hidden_dim

                    # 使用前向聚合器对前向隐藏层和邻居向量进行聚合
                    fw_hidden, fw_hidden_dim = fw_aggregator((fw_hidden, neigh_vec_hidden))

                    # 如果需要保留中间状态，则将前向隐藏层和前向隐藏层维度添加到列表中
                    if keep_inter_state:
                        inter_fw_hiddens.append(fw_hidden)
                        inter_dims.append(fw_hidden_dim)

                    # 如果图编码方向为双向
                    if self.graph_encode_direction == "bi":
                        # 如果跳数为6，则使用第6个后向聚合器
                        if hop > 6:
                            bw_aggregator = bw_aggregators[layer][6]
                        # 如果跳数小于后向聚合器的数量，则使用对应跳数的后向聚合器
                        elif len(bw_aggregators[layer]) > hop:
                            bw_aggregator = bw_aggregators[layer][hop]
                        # 否则，根据聚合类型创建后向聚合器
                        else:
                            if agg_type == "GCN":
                                bw_aggregator = GCNAggregator(bw_hidden_dim, output_node_dim, concat=self.concat,
                                                              dropout=self.dropout, mode=self.mode)
                            elif agg_type == "mean_pooling":
                                bw_aggregator = MeanAggregator(bw_hidden_dim, output_node_dim, concat=self.concat,
                                                               dropout=self.dropout, if_use_high_way=self.with_gcn_highway, mode=self.mode)
                            elif agg_type == "max_pooling":
                                bw_aggregator = MaxPoolingAggregator(bw_hidden_dim, output_node_dim, concat=self.concat,
                                                                     dropout=self.dropout, mode=self.mode)
                            elif agg_type == "lstm":
                                bw_aggregator = SeqAggregator(bw_hidden_dim, output_node_dim, concat=self.concat,
                                                              dropout=self.dropout, mode=self.mode)
                            elif agg_type == "att":
                                bw_aggregator = AttentionAggregator(bw_hidden_dim, output_node_dim, concat=self.concat,
                                                                    mode=self.mode, dropout=self.dropout)

                            # 将后向聚合器添加到后向聚合器列表中
                            bw_aggregators[layer].append(bw_aggregator)

                        # 如果层为0且跳数为0，则使用后向采样邻居的嵌入表示
                        if layer == 0 and hop == 0:
                            neigh_vec_hidden = tf.nn.embedding_lookup(embedded_node_rep, bw_sampled_neighbors)
                        # 否则，使用前向隐藏层和全零向量拼接的嵌入表示
                        else:
                            neigh_vec_hidden = tf.nn.embedding_lookup(
                                tf.concat([bw_hidden, tf.zeros([1, fw_hidden_dim])], 0), bw_sampled_neighbors)

                        # 如果使用GCN高路，则使用多高路层对后向隐藏层进行变换
                        if self.with_gcn_highway:
                            with tf.variable_scope("bw_hidden_highway"):
                                bw_hidden = multi_highway_layer(bw_hidden, fw_hidden_dim, options['highway_layer_num'])

                        # 使用后向聚合器对后向隐藏层和邻居向量进行聚合
                        bw_hidden, bw_hidden_dim = bw_aggregator((bw_hidden, neigh_vec_hidden))

                        if keep_inter_state:
                            inter_bw_hiddens.append(bw_hidden)

            node_dim = fw_hidden_dim

            # hidden stores the representation for all nodes
            # 执行特征聚合
            fw_hidden = tf.reshape(fw_hidden, [-1, single_graph_nodes_size, node_dim])
            # 双向图处理分支
            # 如果图编码方向为双向
            if self.graph_encode_direction == "bi":
                # 将反向隐藏状态进行重塑
                bw_hidden = tf.reshape(bw_hidden, [-1, single_graph_nodes_size, node_dim])
                # 将正向隐藏状态和反向隐藏状态进行拼接
                hidden = tf.concat([fw_hidden, bw_hidden], axis=2)
                # 图的维度为2倍的节点维度
                graph_dim = 2 * node_dim
            else:
                # 如果图编码方向为单向，则隐藏状态为正向隐藏状态
                hidden = fw_hidden
                # 图的维度为节点维度
                graph_dim = node_dim

            # 对hidden进行ReLU激活
            hidden = tf.nn.relu(hidden)
            # 对hidden进行最大池化操作，得到max_pooled
            max_pooled = tf.reduce_max(hidden, 1)
            # 对hidden进行平均池化操作，得到mean_pooled
            mean_pooled = tf.reduce_mean(hidden, 1)
            # 将hidden添加到res列表中
            res = [hidden]

            # 将max_pooled重新塑形为[-1, graph_dim]的形状
            max_graph_embedding = tf.reshape(max_pooled, [-1, graph_dim])
            # 将mean_pooled重新塑形为[-1, graph_dim]的形状
            mean_graph_embedding = tf.reshape(mean_pooled, [-1, graph_dim])
            # 将max_graph_embedding添加到res列表中
            res.append(max_graph_embedding)
            # 将mean_graph_embedding添加到res列表中
            res.append(mean_graph_embedding)
            # 将graph_dim添加到res列表中
            res.append(graph_dim)

            # 中间状态处理分支
            if keep_inter_state:
                # 各中间层状态重塑与池化（处理逻辑与最终层类似）
                inter_node_reps = []
                inter_graph_reps = []
                inter_graph_dims = []
                # process the inter hidden states
                # 遍历inter_fw_hiddens列表
                for _ in range(len(inter_fw_hiddens)):
                    # 获取inter_fw_hiddens列表中的第_个元素
                    inter_fw_hidden = inter_fw_hiddens[_]
                    # 获取inter_bw_hiddens列表中的第_个元素
                    inter_bw_hidden = inter_bw_hiddens[_]
                    # 获取inter_dims列表中的第_个元素
                    inter_dim = inter_dims[_]
                    # 将inter_fw_hidden重塑为[-1, single_graph_nodes_size, inter_dim]的形状
                    inter_fw_hidden = tf.reshape(inter_fw_hidden, [-1, single_graph_nodes_size, inter_dim])

                    # 如果graph_encode_direction为"bi"，则将inter_bw_hidden重塑为[-1, single_graph_nodes_size, inter_dim]的形状
                    if self.graph_encode_direction == "bi":
                        inter_bw_hidden = tf.reshape(inter_bw_hidden, [-1, single_graph_nodes_size, inter_dim])
                        # 将inter_fw_hidden和inter_bw_hidden在axis=2上拼接
                        inter_hidden = tf.concat([inter_fw_hidden, inter_bw_hidden], axis=2)
                        # inter_graph_dim为inter_dim的2倍
                        inter_graph_dim = inter_dim * 2
                    else:
                        # 否则，inter_hidden为inter_fw_hidden
                        inter_hidden = inter_fw_hidden
                        # inter_graph_dim为inter_dim
                        inter_graph_dim = inter_dim

                    # 对inter_hidden进行relu激活
                    inter_node_rep = tf.nn.relu(inter_hidden)
                    # 将inter_node_rep添加到inter_node_reps列表中
                    inter_node_reps.append(inter_node_rep)
                    # 将inter_graph_dim添加到inter_graph_dims列表中
                    inter_graph_dims.append(inter_graph_dim)

                    # 对inter_node_rep在axis=1上进行最大池化
                    max_pooled_tmp = tf.reduce_max(inter_node_rep, 1)
                    mean_pooled_tmp = tf.reduce_max(inter_node_rep, 1)
                    max_graph_embedding = tf.reshape(max_pooled_tmp, [-1, inter_graph_dim])
                    mean_graph_embedding = tf.reshape(mean_pooled_tmp, [-1, inter_graph_dim])
                    inter_graph_reps.append((max_graph_embedding, mean_graph_embedding))

                res.append(inter_node_reps)
                res.append(inter_graph_reps)
                res.append(inter_graph_dims)

            return res

    def act(self, sess, mode, dict, if_pred_on_dev):
        # 设置是否在开发集上进行预测
        self.if_pred_on_dev = if_pred_on_dev

        # 构建输入字典
        # 创建一个字典，用于存储输入数据
        feed_dict = {
            # 将y_true赋值为dict中的y
            self.y_true : np.array(dict['y']),
            # 将fw_adj_info_first赋值为dict中的fw_adj_info_first
            self.fw_adj_info_first : np.array(dict['fw_adj_info_first']),
            self.bw_adj_info_first : np.array(dict['bw_adj_info_first']),
            self.feature_info_first : np.array(dict['feature_info_first']),
            self.feature_len_first : np.array(dict['feature_len_first']),
            self.batch_nodes_first : np.array(dict['batch_nodes_first']),
            self.batch_mask_first : np.array(dict['batch_mask_first']),
            self.looking_table_first : np.array(dict['looking_table_first']),

            self.fw_adj_info_second : np.array(dict['fw_adj_info_second']),
            self.bw_adj_info_second : np.array(dict['bw_adj_info_second']),
            self.feature_info_second : np.array(dict['feature_info_second']),
            self.feature_len_second : np.array(dict['feature_len_second']),
            self.batch_nodes_second : np.array(dict['batch_nodes_second']),
            self.batch_mask_second : np.array(dict['batch_mask_second']),
            self.looking_table_second : np.array(dict['looking_table_second']),
        }

        # 根据模式选择输出
        if mode == "train" and not if_pred_on_dev:
            # 训练模式，不进行预测
            output_feeds = [self.watch, self.training_op, self.loss]
        elif mode == "test" or if_pred_on_dev:
            # 测试模式或进行预测
            output_feeds = [self.y_pred]

        # 运行会话
        results = sess.run(output_feeds, feed_dict)
        return results

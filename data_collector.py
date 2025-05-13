import numpy as np
import configure as conf
from tqdm import tqdm
import DBP15K.preprocessor as preprocessor

def read_data(input_path, graph_dir_name, word_idx, if_increase_dct):
    # 定义g1_triple_path和g1_feature_path的路径
    g1_triple_path = graph_dir_name + "triples_1"
    g1_feature_path = graph_dir_name + "id_features_1"
    # 定义g2_triple_path和g2_feature_path的路径
    g2_triple_path = graph_dir_name + "triples_2"
    g2_feature_path = graph_dir_name + "id_features_2"

    # 生成g1_map和g2_map
    g1_map = preprocessor.gen_graph(g1_triple_path, g1_feature_path)
    g2_map = preprocessor.gen_graph(g2_triple_path, g2_feature_path)

    # 初始化graphs_1、graphs_2和labels
    graphs_1 = []
    graphs_2 = []
    labels = []
    # 打开input_path文件
    with open(input_path, 'r') as fr:
        # 读取文件中的所有行
        lines = fr.readlines()
        # 遍历每一行
        for _ in tqdm(range(len(lines))):
            # 去除行首行尾的空格
            line = lines[_].strip()
            # 将行按照制表符分割
            info = line.split("\t")
            # 将id_1转换为整数
            id_1 = int(info[0])
            # 将id_2转换为整数
            id_2 = int(info[1])
            # 将label转换为整数
            label = int(info[2])

            # 获取g1_map中id_1对应的图
            graph_1 = g1_map[id_1]
            # 获取g2_map中id_2对应的图
            graph_2 = g2_map[id_2]

            # 将g1_map和g2_map中的图添加g_id属性
            graph_1['g_id'] = id_1
            graph_2['g_id'] = id_2

            # 将graph_1、graph_2和label添加到对应的列表中
            graphs_1.append(graph_1)
            graphs_2.append(graph_2)
            labels.append(label)

            # 如果if_increase_dct为True，则更新word_idx
            if if_increase_dct:
                # 获取graph_1和graph_2中的g_ids_features
                features = [graph_1['g_ids_features'], graph_2['g_ids_features']]
                # 遍历features中的每一个特征
                for f in features:
                    # 遍历特征中的每一个id
                    for id in f:
                        # 将特征中的id按照空格分割
                        for w in f[id].split():
                            # 如果w不在word_idx中，则将其添加到word_idx中
                            if w not in word_idx:
                                word_idx[w] = len(word_idx) + 1

    # 返回graphs_1、graphs_2和labels
    return graphs_1, graphs_2, labels



# 定义一个函数，将文本数据向量化
def vectorize_data(word_idx, texts):
    # 创建一个空列表，用于存储向量化后的数据
    tv = []
    # 遍历文本数据
    for text in texts:
        # 创建一个空列表，用于存储每个文本的向量化数据
        stv = []
        # 遍历每个文本的单词
        for w in text.split():
            # 如果单词不在单词索引中，则将未知单词的索引添加到stv中
            if w not in word_idx:
                stv.append(word_idx[conf.unknown_word])
            # 否则，将单词的索引添加到stv中
            else:
                stv.append(word_idx[w])
        # 将每个文本的向量化数据添加到tv中
        tv.append(stv)
    # 返回向量化后的数据
    return tv

def batch_graph(graphs):
    # 定义一个字典，用于存储每个图的节点特征
    g_ids_features = {}
    # 定义一个字典，用于存储每个图的正向邻接矩阵
    g_fw_adj = {}
    # 定义一个字典，用于存储每个图的反向邻接矩阵
    g_bw_adj = {}
    # 定义一个列表，用于存储每个图的节点
    g_nodes = []

    # 遍历每个图
    for g in graphs:
        # 获取每个图的邻接矩阵和节点特征
        id_adj = g['g_adj']
        features = g['g_ids_features']

        # 定义一个列表，用于存储每个图的节点
        nodes = []

        # we first add all nodes into batch_graph and create a mapping from graph id to batch_graph id, this mapping will be
        # used in the creation of fw_adj and bw_adj

        # 创建一个空字典，用于存储id和gid的映射关系
        id_gid_map = {}
        # 获取g_ids_features字典的长度，作为偏移量
        offset = len(g_ids_features.keys())
        # 遍历features字典中的每一个id
        for id in features.keys():
            # 将id转换为整数类型
            id = int(id)
            # 将features字典中的id对应的特征值存储到g_ids_features字典中，gid为偏移量加上id
            g_ids_features[offset + id] = features[id]
            # 将id和gid的映射关系存储到id_gid_map字典中
            id_gid_map[id] = offset + id
            # 将gid存储到nodes列表中
            nodes.append(offset + id)
        # 将nodes列表存储到g_nodes列表中
        g_nodes.append(nodes)

        # 遍历id_adj中的每个id
        for id in id_adj:
            # 获取id对应的邻接表
            adj = id_adj[id]
            # 将id转换为整数
            id = int(id)
            # 获取id对应的图id
            g_id = id_gid_map[id]
            # 如果图id不在g_fw_adj中，则创建一个空列表
            if g_id not in g_fw_adj:
                g_fw_adj[g_id] = []
            # 遍历邻接表中的每个t
            for t in adj:
                # 将t转换为整数
                t = int(t)
                # 获取t对应的图id
                g_t = id_gid_map[t]
                # 将t对应的图id添加到g_fw_adj中
                g_fw_adj[g_id].append(g_t)
                # 如果t对应的图id不在g_bw_adj中，则创建一个空列表
                if g_t not in g_bw_adj:
                    g_bw_adj[g_t] = []
                # 将id对应的图id添加到g_bw_adj中
                g_bw_adj[g_t].append(g_id)

    # 获取节点数量
    node_size = len(g_ids_features.keys())
    # 遍历节点
    for id in range(node_size):
        # 如果节点不在前向邻接表中，则添加一个空列表
        if id not in g_fw_adj:
            g_fw_adj[id] = []
        # 如果节点不在后向邻接表中，则添加一个空列表
        if id not in g_bw_adj:
            g_bw_adj[id] = []

    # 创建一个图对象
    graph = {}
    # 将节点特征添加到图中
    graph['g_ids_features'] = g_ids_features
    # 将节点添加到图中
    graph['g_nodes'] = g_nodes
    # 将前向邻接表添加到图中
    graph['g_fw_adj'] = g_fw_adj
    # 将后向邻接表添加到图中
    graph['g_bw_adj'] = g_bw_adj

    return graph

def vectorize_label(labels):
    # 定义一个空列表lv
    lv = []
    # 遍历labels中的每一个元素
    for label in labels:
        # 如果label为0或'0'，则将[1, 0]添加到lv中
        if label == 0 or label == '0':
            lv.append([1, 0])
        # 如果label为1或'1'，则将[0, 1]添加到lv中
        elif label == 1 or label == '1':
            lv.append([0, 1])
        # 如果label既不是0也不是1，则打印错误信息
        else:
            print("error in vectoring the label")
    # 将lv转换为numpy数组
    lv = np.array(lv)
    # 返回lv
    return lv


def vectorize_batch_graph(graph, word_idx):
    # vectorize the graph feature and normalize the adj info
    id_features = graph['g_ids_features']
    gv = {}
    nv = []
    n_len_v = []
    word_max_len = 0
    # 遍历id_features中的每个id
    for id in id_features:
        # 获取id对应的feature
        feature = id_features[id]
        # 更新word_max_len为feature中单词的最大长度
        word_max_len = max(word_max_len, len(feature.split()))
    # word_max_len = min(word_max_len, conf.word_size_max)

    # 遍历graph中的g_ids_features
    for id in graph['g_ids_features']:
        # 获取feature
        feature = graph['g_ids_features'][id]
        # 初始化fv
        fv = []
        # 遍历feature中的每个token
        for token in feature.split():
            # 如果token为空，则跳过
            if len(token) == 0:
                continue
            # 如果token在word_idx中，则将对应的索引添加到fv中
            if token in word_idx:
                fv.append(word_idx[token])
            # 否则，将unknown_word的索引添加到fv中
            else:
                fv.append(word_idx[conf.unknown_word])

        # 如果fv的长度大于word_max_len，则将word_max_len添加到n_len_v中
        if len(fv) > word_max_len:
            n_len_v.append(word_max_len)
        # 否则，将fv的长度添加到n_len_v中
        else:
            n_len_v.append(len(fv))

        # 如果fv的长度小于word_max_len，则用0填充fv
        for _ in range(word_max_len - len(fv)):
            fv.append(0)
        # 将fv截断为word_max_len
        fv = fv[:word_max_len]
        # 将fv添加到nv中
        nv.append(fv)

    # add an all-zero vector for the PAD node
    # 创建一个长度为word_max_len的列表，每个元素都为0，并添加到nv中
    nv.append([0 for temp in range(word_max_len)])
    # 将n_len_v中的元素加1
    n_len_v.append(0)

    # 将nv转换为numpy数组，并赋值给gv['g_ids_features']
    gv['g_ids_features'] = np.array(nv)
    # 将n_len_v转换为numpy数组，并赋值给gv['g_ids_feature_lens']
    gv['g_ids_feature_lens'] = np.array(n_len_v)

    # ============== vectorize adj info ======================
    # 获取前向邻接矩阵
    g_fw_adj = graph['g_fw_adj']
    # 初始化前向邻接矩阵的值
    g_fw_adj_v = []

    # 初始化最大度数
    degree_max_size = 0
    # 遍历前向邻接矩阵
    for id in g_fw_adj:
        # 更新最大度数
        degree_max_size = max(degree_max_size, len(g_fw_adj[id]))
    # 获取后向邻接矩阵
    g_bw_adj = graph['g_bw_adj']
    # 遍历后向邻接矩阵
    for id in g_bw_adj:
        # 更新最大度数
        degree_max_size = max(degree_max_size, len(g_bw_adj[id]))
    # 将最大度数与配置文件中的采样大小进行比较，取较小值
    degree_max_size = min(degree_max_size, conf.sample_size_per_layer)

    # 遍历前向邻接矩阵
    for id in g_fw_adj:
        # 获取前向邻接矩阵的值
        adj = g_fw_adj[id]
        # 如果前向邻接矩阵的值小于最大度数，则补充
        for _ in range(degree_max_size - len(adj)):
            adj.append(len(g_fw_adj.keys()))
        # 截取前向邻接矩阵的值，使其长度等于最大度数
        adj = adj[:degree_max_size]
        # 断言前向邻接矩阵的值长度等于最大度数
        assert len(adj) == degree_max_size
        # 将前向邻接矩阵的值添加到前向邻接矩阵的值列表中
        g_fw_adj_v.append(adj)

    # PAD node directs to the PAD node
    # 将g_fw_adj.keys()的长度作为列表，重复degree_max_size次，添加到g_fw_adj_v中
    g_fw_adj_v.append([len(g_fw_adj.keys()) for _ in range(degree_max_size)])

    # 初始化一个空列表g_bw_adj_v
    g_bw_adj_v = []
    # 遍历g_bw_adj中的每个id
    for id in g_bw_adj:
        # 获取id对应的邻接表adj
        adj = g_bw_adj[id]
        # 如果adj的长度小于degree_max_size，则向adj中添加len(g_bw_adj.keys())，直到adj的长度等于degree_max_size
        for _ in range(degree_max_size - len(adj)):
            adj.append(len(g_bw_adj.keys()))
        # 将adj截取为degree_max_size长度
        adj = adj[:degree_max_size]
        # 断言adj的长度等于degree_max_size
        assert len(adj) == degree_max_size
        # 将adj添加到g_bw_adj_v中
        g_bw_adj_v.append(adj)

    # PAD node directs to the PAD node
    g_bw_adj_v.append([len(g_bw_adj.keys()) for _ in range(degree_max_size)])

    # ============== vectorize nodes info ====================
    # 获取图中的节点
    g_nodes = graph['g_nodes']
    # 初始化图的最大节点数
    graph_max_size = 0
    # 遍历图中的节点
    for nodes in g_nodes:
        # 更新图的最大节点数
        graph_max_size = max(graph_max_size, len(nodes))

    # 初始化节点值、节点掩码和实体索引
    g_node_v = []
    g_node_mask = []
    entity_index = []
    # 遍历图中的节点
    for nodes in g_nodes:
        # 初始化节点掩码
        mask = [1 for _ in range(len(nodes))]
        # 如果节点数小于图的最大节点数，则补充节点和掩码
        for _ in range(graph_max_size - len(nodes)):
            nodes.append(len(g_fw_adj.keys()))
            mask.append(0)
        # 截取节点和掩码到图的最大节点数
        nodes = nodes[:graph_max_size]
        mask = mask[:graph_max_size]
        # 将节点和掩码添加到节点值和节点掩码列表中
        g_node_v.append(nodes)
        g_node_mask.append(mask)
        # 将实体索引添加到实体索引列表中
        entity_index.append(0)

    # 定义一个空列表，用于存储节点索引
    g_looking_table = []
    # 定义一个全局计数器，用于记录节点索引
    global_count = 0
    # 遍历节点掩码
    for mask in g_node_mask:
        # 遍历掩码中的每个元素
        for item in mask:
            # 如果元素为1，则将全局计数器添加到节点索引列表中
            if item == 1:
                g_looking_table.append(global_count)
            # 全局计数器加1
            global_count += 1

    # 将节点向量转换为numpy数组
    gv['g_nodes'] =np.array(g_node_v)
    # 将带宽邻接矩阵转换为numpy数组
    gv['g_bw_adj'] = np.array(g_bw_adj_v)
    # 将前向邻接矩阵转换为numpy数组
    gv['g_fw_adj'] = np.array(g_fw_adj_v)
    # 将节点掩码转换为numpy数组
    gv['g_mask'] = np.array(g_node_mask)
    # 将节点索引列表转换为numpy数组
    gv['g_looking_table'] = np.array(g_looking_table)
    # 将实体索引添加到gv字典中
    gv['entity_index'] = entity_index

    # 返回gv字典
    return gv

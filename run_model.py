import os
import configure as conf
import data_collector as data_collector
import loaderAndwriter as disk_helper
import argparse
import tensorflow as tf
from model import GraphMatchNN
import numpy as np
from tqdm import tqdm
import datetime

def main():
    # 创建一个字典，用于存储单词和索引的对应关系
    word_idx = {}
    # 获取模型类型
    model_type = conf.model_type
    # 获取训练轮数
    epochs = conf.epochs
    # 预训练词向量的大小
    pretrained_word_size = 0
    # 预训练词向量
    pretrained_word_embeddings = np.array([])
    # 如果使用预训练词向量
    if conf.if_use_pretrained_embedding:
        # 打印加载预训练词向量的提示信息
        print("loading pretrained embedding ...")
        # 加载预训练词向量
        pretrained_word_embeddings = disk_helper.load_word_embedding(conf.pretrained_word_embedding_path, word_idx)
        # 获取预训练词向量的大小
        pretrained_word_size = len(pretrained_word_embeddings)
        # 设置隐藏层维度
        conf.hidden_layer_dim = conf.pretrained_word_embedding_dim

        # 打印加载的预训练词向量数量
        print("load {} pre-trained word embeddings from Glove".format(pretrained_word_size))

    # 将未知词的索引设置为字典中单词数量的下一个索引
    word_idx[conf.unknown_word] = len(word_idx.keys()) + 1

    # 设置词索引文件路径
    conf.word_idx_file_path = "saved_model/" + conf.model_name + "/" + conf.word_idx_file_path
    # 设置预测文件路径
    conf.pred_file_path = "saved_model/" + conf.model_name + "/" + conf.pred_file_path

    if model_type == "train":

        # 设置随机种子
        np.random.seed(0)
        # 获取训练批次大小
        train_batch_size = conf.train_batch_size

        # 打印读取训练数据到内存中
        print("reading training data into the mem ...")

        # 读取训练数据
        graphs_1_train, graphs_2_train, labels_train = data_collector.read_data(conf.train_data_path, conf.graph_dir_name, word_idx, True)

        # 打印读取开发数据到内存中
        print("reading development data into the mem ...")
        # 读取开发数据
        graphs_1_dev, graphs_2_dev, labels_dev = data_collector.read_data(conf.dev_data_path, conf.graph_dir_name, word_idx, False)

        # 打印写入词-索引映射
        print("writing word-idx mapping ...")
        # 写入词-索引映射
        disk_helper.write_word_idx(word_idx, conf.word_idx_file_path)

        # 设置词表大小
        conf.word_vocab_size = len(word_idx)
        # 设置预训练词表大小
        conf.pretrained_word_size = pretrained_word_size
        # 设置学习词表大小
        conf.learned_word_size = len(word_idx) - pretrained_word_size

        with tf.Graph().as_default():
            # tf.set_random_seed(0)
            with tf.Session() as sess:
                model = GraphMatchNN("train", conf, pretrained_word_embeddings)
                model._build_graph()

                saver = tf.train.Saver(max_to_keep=None)
                sess.run(tf.initialize_all_variables())

                # 定义训练步骤
                def train_step(g1_v_batch, g2_v_batch, label_v_batch, if_pred_on_dev=False):
                    # 创建一个字典，用于存储输入数据
                    dict = {}
                    # 将第一个图的邻接信息、特征信息、节点信息、掩码信息和查找表信息存储到字典中
                    dict['fw_adj_info_first'] = g1_v_batch['g_fw_adj']
                    dict['bw_adj_info_first'] = g1_v_batch['g_bw_adj']
                    dict['feature_info_first'] = g1_v_batch['g_ids_features']
                    dict['feature_len_first'] = g1_v_batch['g_ids_feature_lens']
                    dict['batch_nodes_first'] = g1_v_batch['g_nodes']
                    dict['batch_mask_first'] = g1_v_batch['g_mask']
                    dict['looking_table_first'] = g1_v_batch['g_looking_table']

                    # 将第二个图的邻接信息、特征信息、节点信息、掩码信息和查找表信息存储到字典中
                    dict['fw_adj_info_second'] = g2_v_batch['g_fw_adj']
                    dict['bw_adj_info_second'] = g2_v_batch['g_bw_adj']
                    dict['feature_info_second'] = g2_v_batch['g_ids_features']
                    dict['feature_len_second'] = g2_v_batch['g_ids_feature_lens']
                    dict['batch_nodes_second'] = g2_v_batch['g_nodes']
                    dict['batch_mask_second'] = g2_v_batch['g_mask']
                    dict['looking_table_second'] = g2_v_batch['g_looking_table']

                    # 将标签信息存储到字典中
                    dict['y'] = label_v_batch

                    # 如果不是在开发集上进行预测，则进行训练
                    if not if_pred_on_dev:
                        # 调用模型的act方法进行训练，并返回损失值
                        watch, _, loss = model.act(sess, "train", dict, if_pred_on_dev)
                        return loss

                    # 如果是在开发集上进行预测，则进行预测
                    else:
                        # 调用模型的act方法进行预测，并返回预测结果
                        predicted = model.act(sess, "train", dict, if_pred_on_dev)
                        return predicted

                # 初始化最佳准确率为0.0
                best_acc = 0.0
                # 遍历所有训练轮次
                for t in range(1, epochs + 1):
                    # 获取训练集长度
                    n_train = len(graphs_1_train)
                    # 生成一个0到n_train-1的列表
                    temp_order = list(range(n_train))
                    # 随机打乱列表
                    np.random.shuffle(temp_order)

                    # 初始化损失和为0.0
                    loss_sum = 0.0
                    # 遍历训练集，每次取一个batch
                    for start in tqdm(range(0, n_train, train_batch_size)):
                        # 计算当前batch的结束位置
                        end = min(start + train_batch_size, n_train)
                        # 初始化当前batch的图、标签列表
                        graphs_1 = []
                        graphs_2 = []
                        labels = []
                        # 遍历当前batch，将图、标签添加到列表中
                        for _ in range(start, end):
                            idx = temp_order[_]
                            graphs_1.append(graphs_1_train[idx])
                            graphs_2.append(graphs_2_train[idx])
                            labels.append(labels_train[idx])

                        # 将图列表转换为batch图
                        batch_graph_1 = data_collector.batch_graph(graphs_1)
                        batch_graph_2 = data_collector.batch_graph(graphs_2)

                        # 将batch图向量化
                        g1_v_batch = data_collector.vectorize_batch_graph(batch_graph_1, word_idx)
                        g2_v_batch = data_collector.vectorize_batch_graph(batch_graph_2, word_idx)
                        # 将标签向量化
                        label_v_batch = data_collector.vectorize_label(labels)

                        # 计算当前batch的损失
                        train_loss = train_step(g1_v_batch, g2_v_batch, label_v_batch, if_pred_on_dev=False)

                        # 将当前batch的损失累加到总损失中
                        loss_sum += train_loss

                    #####################  evaluate the model on the dev data #######################
                    print("evaluating the model on the dev data ...")
                    # 计算开发集数据的大小
                    n_dev = len(graphs_1_dev)
                    # 设置开发集的批处理大小
                    dev_batch_size = conf.dev_batch_size
                    # 初始化金标和预测结果列表
                    golds = []
                    predicted_res = []
                    g1_ori_ids = []
                    g2_ori_ids = []
                    # 遍历开发集数据，每次取一个批处理大小的数据
                    for start in tqdm(range(0, n_dev, dev_batch_size)):
                        end = min(start + dev_batch_size, n_dev)
                        graphs_1 = []
                        graphs_2 = []
                        labels = []
                        # 将当前批处理大小的数据添加到列表中
                        # 遍历从start到end的索引
                        for _ in range(start, end):
                            # 将graphs_1_dev中的元素添加到graphs_1中
                            graphs_1.append(graphs_1_dev[_])
                            graphs_2.append(graphs_2_dev[_])
                            labels.append(labels_dev[_])
                            golds.append(labels_dev[_])

                            g1_ori_ids.append(graphs_1_dev[_]['g_id'])
                            g2_ori_ids.append(graphs_2_dev[_]['g_id'])

                        # 将当前批处理大小的数据转换为张量
                        batch_graph_1 = data_collector.batch_graph(graphs_1)
                        batch_graph_2 = data_collector.batch_graph(graphs_2)

                        g1_v_batch = data_collector.vectorize_batch_graph(batch_graph_1, word_idx)
                        g2_v_batch = data_collector.vectorize_batch_graph(batch_graph_2, word_idx)
                        label_v_batch = data_collector.vectorize_label(labels)

                        # 使用训练步骤对当前批处理大小的数据进行预测
                        predicted = train_step(g1_v_batch, g2_v_batch, label_v_batch, if_pred_on_dev=True)[0]

                        # 将预测结果添加到列表中
                        for _ in range(0, end - start):
                            predicted_res.append(predicted[_][1])  # add the prediction result into the bag

                    count = 0.0
                    # 计数器，用于记录预测结果的数量
                    correct_10 = 0.0
                    # 记录预测结果中前10个中正确的数量
                    correct_1 = 0.0
                    # 记录预测结果中第一个正确的数量
                    cand_size = conf.dev_cand_size
                    # 获取预测结果的候选集大小
                    assert len(predicted_res) % cand_size == 0
                    # 断言预测结果的数量是候选集大小的整数倍
                    assert len(predicted_res) == len(g1_ori_ids)
                    # 断言预测结果的数量和g1_ori_ids的数量相等
                    assert len(g1_ori_ids) == len(g2_ori_ids)
                    # 断言g1_ori_ids的数量和g2_ori_ids的数量相等
                    number = int(len(predicted_res)/cand_size)
                    # 计算预测结果的数量除以候选集大小，得到预测结果的组数
                    incorrect_pairs = []
                    # 用于存储预测结果中前10个中错误的id对
                    for _ in range(number):
                        idx_score = {}
                        # 用于存储每个预测结果的索引和得分
                        for idx in range(cand_size):
                            idx_score[ _ * cand_size + idx ] = predicted_res[ _ * cand_size + idx ]
                        # 将每个预测结果的索引和得分存储到idx_score字典中
                        idx_score_items = idx_score.items()
                        # 将idx_score字典转换为列表
                        idx_score_items = sorted(idx_score_items, key=lambda d: d[1], reverse=True)

                        # 将idx_score列表按照得分从高到低排序
                        id_1 = g1_ori_ids[_ * cand_size]
                        id_2 = g2_ori_ids[_ * cand_size]

                        # 获取当前组中g1_ori_ids和g2_ori_ids的id
                        for sub_idx in range(min(10, len(idx_score_items))):
                            idx = idx_score_items[sub_idx][0]
                            # 获取当前组中前10个预测结果的索引
                            if golds[idx] == 1:
                                correct_10 += 1.0
                                # 如果当前预测结果在gold中存在，则正确数量加1
                                if sub_idx == 0:
                                    correct_1 += 1.0
                                    # 如果当前预测结果是第一个，则正确数量加1
                                else:
                                    incorrect_pairs.append((id_1, id_2))
                                    # 如果当前预测结果不是第一个，则将id对添加到incorrect_pairs中
                                break
                        count += 1.0

                        # 每处理一组预测结果，计数器加1
                    acc_10 = correct_10 / count
                    # 计算前10个预测结果中正确的数量占总数量的比例
                    acc_1 = correct_1 / count

                    # 如果acc_1大于best_acc，则更新best_acc
                    if acc_1 > best_acc:
                        best_acc = acc_1
                        # 保存模型路径
                        save_path = "saved_model/" + conf.model_name + "/"
                        # 如果路径不存在，则创建路径
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        # 保存模型
                        path = saver.save(sess, save_path + 'model', global_step=0)
                        print("Already saved model to {}".format(path))

                        # 写入预测文件
                        print('writing prediction file...')
                        with open(conf.pred_file_path, 'w') as f:
                            # 遍历incorrect_pairs
                            for (id_1, id_2) in incorrect_pairs:
                                # 写入id_1和id_2
                                f.write(str(id_1)+"\t"+str(id_2)+"\n")

                    time_str = datetime.datetime.now().isoformat()
                    print('-----------------------')
                    print('time:{}'.format(time_str))
                    print('Epoch', t)
                    print('Loss on train:{}'.format(loss_sum))
                    print('acc @1 on Dev:{}'.format(acc_1))
                    print('acc @10 on Dev:{}'.format(acc_10))
                    print('best acc @1 on Dev:{}'.format(best_acc))
                    print('-----------------------')

    if model_type == "test":

        # 从文件中读取单词索引映射
        print("reading word idx mapping from file ...")
        word_idx = disk_helper.read_word_idx_from_file(conf.word_idx_file_path)

        # 将训练数据读入内存
        print("reading training data into the mem ...")
        graphs_1_test, graphs_2_test, labels_test = data_collector.read_data(conf.test_data_path, conf.graph_dir_name, word_idx, False)

        # 设置单词词汇表大小
        conf.word_vocab_size = len(word_idx)
        # 设置预训练单词大小
        conf.pretrained_word_size = pretrained_word_size
        # 设置学习单词大小
        conf.learned_word_size = len(word_idx) - pretrained_word_size

        # 创建一个新的计算图
        with tf.Graph().as_default():
            # 创建一个会话
            with tf.Session() as sess:
                # 创建一个GraphMatchNN模型
                model = GraphMatchNN("test", conf, pretrained_word_embeddings)
                # 构建模型图
                model._build_graph()
                # 创建一个保存模型的saver
                saver = tf.train.Saver(max_to_keep=None)

                # 模型保存路径
                model_path_name = "saved_model/" + conf.model_name + "/model-0"
                # 模型预测结果保存路径
                model_pred_path = "saved_model/" + conf.model_name + "/prediction.txt"

                # 从模型保存路径中恢复模型
                saver.restore(sess, model_path_name)

                def test_step(g1_v_batch, g2_v_batch, label_v_batch):
                    # 创建一个字典，用于存储测试数据
                    dict = {}
                    # 将g1_v_batch中的g_fw_adj存入字典
                    dict['fw_adj_info_first'] = g1_v_batch['g_fw_adj']
                    dict['bw_adj_info_first'] = g1_v_batch['g_bw_adj']
                    dict['feature_info_first'] = g1_v_batch['g_ids_features']
                    dict['feature_len_first'] = g1_v_batch['g_ids_feature_lens']
                    dict['batch_nodes_first'] = g1_v_batch['g_nodes']
                    dict['batch_mask_first'] = g1_v_batch['g_mask']
                    dict['looking_table_first'] = g1_v_batch['g_looking_table']
                    dict['entity_index_first'] = g1_v_batch['entity_index']

                    # 将g2_v_batch中的g_fw_adj存入字典
                    dict['fw_adj_info_second'] = g2_v_batch['g_fw_adj']
                    dict['bw_adj_info_second'] = g2_v_batch['g_bw_adj']
                    dict['feature_info_second'] = g2_v_batch['g_ids_features']
                    dict['feature_len_second'] = g2_v_batch['g_ids_feature_lens']
                    dict['batch_nodes_second'] = g2_v_batch['g_nodes']
                    dict['batch_mask_second'] = g2_v_batch['g_mask']
                    dict['looking_table_second'] = g2_v_batch['g_looking_table']
                    dict['entity_index_second'] = g2_v_batch['entity_index']

                    # 将label_v_batch存入字典
                    dict['y'] = label_v_batch
                    # 调用model.act方法，传入sess、"test"、dict和if_pred_on_dev参数，获取预测结果
                    predicted = model.act(sess, "test", dict, if_pred_on_dev=False)
                    # 返回预测结果
                    return predicted

                n_test = len(graphs_1_test)
                # 获取测试集的长度
                test_batch_size = conf.test_batch_size
                # 获取测试集的批处理大小
                golds = []
                # 创建一个空列表，用于存储测试集的真实标签
                predicted_res = []
                # 创建一个空列表，用于存储测试集的预测结果
                g1_ori_ids = []
                # 创建一个空列表，用于存储测试集的第一个图的原始ID
                g2_ori_ids = []
                # 创建一个空列表，用于存储测试集的第二个图的原始ID
                for start in tqdm(range(0, n_test, test_batch_size)):
                    # 遍历测试集，每次取一个批处理大小的数据
                    end = min(start + test_batch_size, n_test)
                    # 计算当前批处理数据的结束位置
                    graphs_1 = []
                    # 创建一个空列表，用于存储当前批处理数据的第一个图
                    graphs_2 = []
                    # 创建一个空列表，用于存储当前批处理数据的第二个图
                    labels = []
                    # 创建一个空列表，用于存储当前批处理数据的标签
                    for _ in range(start, end):
                        # 遍历当前批处理数据
                        graphs_1.append(graphs_1_test[_])
                        # 将当前批处理数据的第一个图添加到列表中
                        graphs_2.append(graphs_2_test[_])
                        # 将当前批处理数据的第二个图添加到列表中
                        labels.append(labels_test[_])
                        # 将当前批处理数据的标签添加到列表中
                        golds.append(labels_test[_])

                        # 将当前批处理数据的标签添加到真实标签列表中
                        g1_ori_ids.append(graphs_1_test[_]['g_id'])
                        # 将当前批处理数据的第一个图的原始ID添加到列表中
                        g2_ori_ids.append(graphs_2_test[_]['g_id'])

                        # 将当前批处理数据的第二个图的原始ID添加到列表中
                    batch_graph_1 = data_collector.batch_graph(graphs_1)
                    # 将当前批处理数据的第一个图转换为批处理图
                    batch_graph_2 = data_collector.batch_graph(graphs_2)

                    # 将当前批处理数据的第二个图转换为批处理图
                    g1_v_batch = data_collector.vectorize_batch_graph(batch_graph_1, word_idx)
                    # 将当前批处理数据的第一个图转换为向量
                    g2_v_batch = data_collector.vectorize_batch_graph(batch_graph_2, word_idx)
                    # 将当前批处理数据的第二个图转换为向量
                    label_v_batch = data_collector.vectorize_label(labels)

                    # 将当前批处理数据的标签转换为向量
                    predicted = test_step(g1_v_batch, g2_v_batch, label_v_batch)[0]

                    # 使用测试步骤对当前批处理数据进行预测
                    for _ in range(0, end - start):
                        # 遍历当前批处理数据的预测结果
                        predicted_res.append(predicted[_][1])  # add the prediction result into the bag

                count = 0.0
                # 初始化正确预测的个数
                correct_10 = 0.0
                # 初始化正确预测的前10个的个数
                correct_1 = 0.0
                # 初始化正确预测的第一个的个数
                cand_size = conf.test_cand_size
                # 获取测试候选集的大小
                assert len(predicted_res) % cand_size == 0
                # 断言预测结果的数量是候选集大小的整数倍
                assert len(predicted_res) == len(g1_ori_ids)
                # 断言预测结果的数量和g1_ori_ids的数量相等
                assert len(g1_ori_ids) == len(g2_ori_ids)
                # 断言g1_ori_ids的数量和g2_ori_ids的数量相等
                number = int(len(predicted_res) / cand_size)
                # 计算预测结果的数量除以候选集大小的商
                incorrect_pairs = []
                # 初始化一个空列表，用于存储错误的预测对
                for _ in range(number):
                    idx_score = {}
                    # 初始化一个空字典，用于存储每个候选集的预测得分
                    for idx in range(cand_size):
                        idx_score[_ * cand_size + idx] = predicted_res[_ * cand_size + idx]
                    # 将每个候选集的预测得分存储到字典中
                    idx_score_items = idx_score.items()
                    # 将字典转换为列表
                    idx_score_items = sorted(idx_score_items, key=lambda d: d[1], reverse=True)

                    id_1 = g1_ori_ids[_ * cand_size]
                    id_2 = g2_ori_ids[_ * cand_size]

                    for sub_idx in range(min(10, len(idx_score_items))):
                        idx = idx_score_items[sub_idx][0]
                        # 获取得分最高的候选集的索引
                        if golds[idx] == 1:
                            # 如果该候选集是正确的
                            correct_10 += 1.0
                            # 正确预测的前10个的个数加1
                            if sub_idx == 0:
                                correct_1 += 1.0
                                # 正确预测的第一个的个数加1
                            else:
                                incorrect_pairs.append((id_1, id_2))
                                # 将错误的预测对添加到列表中
                            break
                    count += 1.0

                # 计算准确率
                acc_10 = correct_10 / count
                acc_1 = correct_1 / count
                # 打印准确率
                print('-----------------------')
                print('acc @1 on Test:{}'.format(acc_1))
                print('acc @10 on Test:{}'.format(acc_10))
                print('-----------------------')
                # 写入预测文件
                print('writing prediction file...')
                with open(conf.pred_file_path, 'w') as f:
                    # 遍历错误对
                    for (id_1, id_2) in incorrect_pairs:
                        # 写入文件
                        f.write(str(id_1) + "\t" + str(id_2) + "\n")


if __name__ == "__main__":
    # 创建一个参数解析器
    argparser = argparse.ArgumentParser()
    # 添加一个参数，用于指定模式，只能是train或test
    argparser.add_argument("mode", type=str, choices=["train", "test"])
    # 添加一个参数，用于指定任务，只能是zh_en、en_zh、fr_en、en_fr、ja_en、en_ja
    argparser.add_argument("task", type=str, choices=["zh_en", "en_zh", "fr_en", "en_fr", "ja_en", "en_ja"])
    # 添加一个参数，用于指定模型的名称
    argparser.add_argument("name", type=str, help=("specify the name of the model"))
    # 添加一个参数，用于指定第一个gcn的窗口大小，默认值为conf.gcn_window_size_first
    argparser.add_argument("-gcn_window_size_first", type=int, default=conf.gcn_window_size_first, help="window size at first gcn")
    # 添加一个参数，用于指定第一个gcn的层数，默认值为conf.gcn_layer_size_first
    argparser.add_argument("-gcn_layer_size_first", type=int, default=conf.gcn_layer_size_first, help="layer size at first gcn")
    # 添加一个参数，用于指定第二个gcn的窗口大小，默认值为conf.gcn_window_size_second
    argparser.add_argument("-gcn_window_size_second", type=int, default=conf.gcn_window_size_second, help="window size at second gcn")
    # 添加一个参数，用于指定第二个gcn的层数，默认值为conf.gcn_layer_size_second
    argparser.add_argument("-gcn_layer_size_second", type=int, default=conf.gcn_layer_size_second, help="layer size at second gcn")
    # 添加一个参数，用于指定第一个gcn节点的表示维度，默认值为conf.aggregator_dim_first
    argparser.add_argument("-aggregator_dim_first", type=int, default=conf.aggregator_dim_first, help="first gcn node rep dim")
    # 添加一个参数，用于指定第二个gcn节点的表示维度，默认值为conf.aggregator_dim_second
    argparser.add_argument("-aggregator_dim_second", type=int, default=conf.aggregator_dim_second, help="second gcn node rep dim")
    # 添加一个参数，用于指定第一个gcn的类型，默认值为conf.gcn_type_first
    argparser.add_argument("-gcn_type_first", type=str, default=conf.gcn_type_first, help = "first gcn type")
    # 添加一个参数，用于指定第二个gcn的类型，默认值为conf.gcn_type_second
    argparser.add_argument("-gcn_type_second", type=str, default=conf.gcn_type_second, help = "second gcn type")
    # 添加一个参数，用于指定每层的采样大小，默认值为conf.sample_size_per_layer
    argparser.add_argument("-sample_size_per_layer", type=int, default=conf.sample_size_per_layer, help="sample size per layer")
    # 添加一个参数，用于指定训练的轮数，默认值为conf.epochs
    argparser.add_argument("-epochs", type=int, default=conf.epochs, help="training epochs")
    # 添加一个参数，用于指定学习率，默认值为conf.learning_rate
    argparser.add_argument("-learning_rate", type=float, default=conf.learning_rate, help="learning rate")
    # 添加一个参数，用于指定隐藏层的维度，默认值为conf.hidden_layer_dim
    argparser.add_argument("-hidden_layer_dim", type=int, default=conf.hidden_layer_dim)
    # 添加一个参数，用于指定是否使用预训练的词向量，默认值为False
    argparser.add_argument("-use_pretrained_embedding", action='store_true', default=False, help="if use glove embedding")
    # 添加一个参数，用于指定是否使用匹配的高way，默认值为False
    argparser.add_argument("-with_match_highway", action='store_true', default=False, help="with match highway")
    # 添加一个参数，用于指定mp的维度，默认值为conf.cosine_MP_dim
    argparser.add_argument("-cosine_MP_dim", type=int, default=conf.cosine_MP_dim, help="mp dim")
    # 添加一个参数，用于指定dropout的比率，默认值为conf.dropout
    argparser.add_argument("-drop_out", type=float, default=conf.dropout, help="dropout rate")
    # 添加一个参数，用于指定预测的方法，只能是graph_level或node_level
    argparser.add_argument("-pred_method", type=str, default=conf.pred_method, choices=['graph_level', 'node_level'])

    # 解析命令行参数
    config = argparser.parse_args()

    # 将命令行参数赋值给配置变量
    # 设置模型类型
    conf.model_type = config.mode
    # 设置第一层GCN的窗口大小
    conf.gcn_window_size_first = config.gcn_window_size_first
    # 设置第二层GCN的窗口大小
    conf.gcn_window_size_second = config.gcn_window_size_second
    # 设置每层采样的样本大小
    conf.sample_size_per_layer = config.sample_size_per_layer
    # 设置训练的轮数
    conf.epochs = config.epochs
    # 设置学习率
    conf.learning_rate = config.learning_rate
    # 设置隐藏层的维度
    conf.hidden_layer_dim = config.hidden_layer_dim
    # 设置第一层聚合器的维度
    conf.aggregator_dim_first = config.aggregator_dim_first
    # 设置第二层聚合器的维度
    conf.aggregator_dim_second = config.aggregator_dim_second
    # 设置第一层GCN的层数
    conf.gcn_layer_size_first = config.gcn_layer_size_first
    # 设置第二层GCN的层数
    conf.gcn_layer_size_second = config.gcn_layer_size_second
    # 设置第一层GCN的类型
    conf.gcn_type_first = config.gcn_type_first
    # 设置第二层GCN的类型
    conf.gcn_type_second = config.gcn_type_second
    # 设置余弦相似度匹配的维度
    conf.cosine_MP_dim = config.cosine_MP_dim
    # 设置dropout率
    conf.dropout = config.drop_out
    # 设置是否使用预训练的嵌入
    conf.if_use_pretrained_embedding = config.use_pretrained_embedding
    # 设置预测方法
    conf.pred_method = config.pred_method
    # 设置任务类型
    conf.task = config.task

    conf.train_data_path = "DBP15K/" + conf.task + "/train.examples." + str(conf.train_cand_size)
    conf.dev_data_path = "DBP15K/" + conf.task + "/dev.examples." + str(conf.dev_cand_size)
    conf.test_data_path = "DBP15K/" + conf.task + "/test.examples." + str(conf.test_cand_size)
    conf.graph_dir_name = "DBP15K/" + conf.task + "/"


    # 如果使用预训练的词嵌入，则将隐藏层维度设置为预训练词嵌入维度
    if conf.if_use_pretrained_embedding:
        conf.hidden_layer_dim = conf.pretrained_word_embedding_dim

    # 根据配置参数生成模型名称
    conf.model_name = config.name + "_win1_" + str(conf.gcn_window_size_first) + "_win2_" + str(conf.gcn_window_size_second) + "_node1dim_" + str(conf.aggregator_dim_first) + "_node2dim_" + str(conf.aggregator_dim_second) \
                       + "_word_embedding_dim_" + str(conf.hidden_layer_dim) + "_layer1_" + str(conf.gcn_layer_size_first) + "_layer2_" + str(conf.gcn_layer_size_second) + "_first_gcn_type_" + conf.gcn_type_first + "_second_gcn_type_" + conf.gcn_type_second \
                       + "_cosine_MP_dim_" + str(conf.cosine_MP_dim) + "_drop_out_" + str(conf.dropout) + "_use_Glove_" + str(conf.if_use_pretrained_embedding) + "_pm_" + conf.pred_method + "_sample_size_per_layer_" + str(conf.sample_size_per_layer)
    # 调用主函数
    main()

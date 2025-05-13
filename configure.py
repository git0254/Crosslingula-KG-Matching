#候选集大小的设置       预测层？
train_cand_size = 20

#开发候选集大小
dev_cand_size = 20

#测试候选集大小
test_cand_size = 1000

# in the future version, the word idx should be directed to the model related dir
word_idx_file_path = "data/word.idx"
pred_file_path = "data/pred.txt"
# label_idx_file_path = "data/label.idx"
#训练的超参数
# 训练集批次大小
train_batch_size = 32
# 验证集批次大小
dev_batch_size = 20
# 测试集批次大小
test_batch_size = 100

# L2正则化参数
l2_lambda = 0.000001
# 学习率
learning_rate = 0.001
# 迭代次数
epochs = 10
# 编码器隐藏层维度
encoder_hidden_dim = 200
# 单词最大长度
word_size_max = 1

dropout = 0.0

node_vec_method = "lstm" # lstm or word_emb 用 LSTM 编码实体名称

# path_embed_method = "lstm" # cnn or lstm or bi-lstm

# 定义未知单词的标记
unknown_word = "**UNK**"
# 是否处理未知单词
deal_unknown_words = True

# 是否使用预训练的词向量
if_use_pretrained_embedding = True
# 预训练词向量的维度
pretrained_word_embedding_dim = 300
# 预训练词向量的路径
pretrained_word_embedding_path = "DBP15K/sub.glove.300d"
# 词向量的维度
word_embedding_dim = 100

num_layers = 1 # 1 or 2

# the following are for the graph encoding method
# 权重衰减
weight_decay = 0.0000
# 每层样本大小
sample_size_per_layer = 1
# 隐藏层维度
hidden_layer_dim = 100
# 特征最大长度
feature_max_len = 1
# 特征编码类型
feature_encode_type = "uni"
# graph_encode_method = "max-pooling" # "lstm" or "max-pooling"
graph_encode_direction = "bi" # "single" or "bi"

concat = True

encoder = "gated_gcn" # "gated_gcn" "gcn"  "seq"

lstm_in_gcn = "none" # before, after, none

# 定义聚合器的第一个维度
aggregator_dim_first = 100
# 定义聚合器的第二个维度
aggregator_dim_second = 100
# 定义第一个GCN的窗口大小
gcn_window_size_first = 1
# 定义第二个GCN的窗口大小
gcn_window_size_second = 2
# 定义第一个GCN的层数
gcn_layer_size_first = 1
# 定义第二个GCN的层数
gcn_layer_size_second = 1

# 是否使用匹配的高way
with_match_highway = False
# 是否使用GCN的高way
with_gcn_highway = False
# 是否使用多个GCN的第一状态
if_use_multiple_gcn_1_state = False
# 是否使用多个GCN的第二状态
if_use_multiple_gcn_2_state = False

# 是否使用多个GCN进行状态预测
agg_sim_method = "GCN" # "GCN" or LSTM

# GCN类型
gcn_type_first = 'mean_pooling' # GCN, max_pooling, mean_pooling, lstm, att
gcn_type_second = 'mean_pooling'

# 余弦相似度维度
cosine_MP_dim = 10

# 预测方法
pred_method = "graph_level"  # graph_level or node_level



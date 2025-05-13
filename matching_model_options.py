options = {
    "aggregation_layer_num": 1, #聚合的层数
    "with_full_match": True,    #启用全匹配方法    即将每个实体与对方主题实体（anchor）进行对齐匹配
    "with_maxpool_match": True, #启用最大池匹配方法  将所有匹配分数中取 最大值 作为特征
    "with_max_attentive_match": True,   #启用最大注意力匹配方法    使用注意力机制，挑选与之 最相关 的对方实体进行匹配
    "with_attentive_match": True,   #启用注意力匹配方法  注意力加权的向量用于匹配
    "with_cosine": True,    #余弦相似度匹配    余弦距离衡量两个嵌入向量的相似程度
    "with_mp_cosine": True, #多重匹配？？
    "highway_layer_num": 1, #网络层数
    "with_highway": True,   #输入表示上使用 Highway 网络
    "with_match_highway": True, #匹配向量之后使用 Highway 网络
    "with_aggregation_highway": True,   #聚合模块（图级匹配）中使用 Highway 网络
    "use_cudnn": False, #
    "aggregation_lstm_dim": 100,     #LSTM 隐藏层维度
    "with_moving_average": False    #
}

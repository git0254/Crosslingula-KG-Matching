import tensorflow as tf
from inits import zeros
import configure as conf

_LAYER_UIDS = {}

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
        Implementation inspired by keras (http://keras.io).
        # Properties
            name: String, defines the variable scope of the layer.
            logging: Boolean, switches Tensorflow histogram logging on/off

        # Methods
            _call(inputs): Defines computation graph of layer
                (i.e. takes input, returns output)
            __call__(inputs): Wrapper for _call()
        """

    def __init__(self, **kwargs):
        # 定义允许的关键字参数
        allowed_kwargs = {'name', 'logging', 'model_size'}
        # 遍历关键字参数
        for kwarg in kwargs.keys():
            # 断言关键字参数是否在允许的关键字参数中
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        # 获取关键字参数name
        name = kwargs.get('name')
        # 如果没有name，则生成一个name
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        # 将name赋值给self.name
        self.name = name
        # 初始化self.vars为空字典
        self.vars = {}
        # 获取关键字参数logging，如果没有则默认为False
        logging = kwargs.get('logging', False)
        # 将logging赋值给self.logging
        self.logging = logging
        # 初始化self.sparse_inputs为False
        self.sparse_inputs = False

    # 定义一个名为_call的方法，接收一个参数inputs
    def _call(self, inputs):
        # 返回参数inputs
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs

class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, dropout=0.,
                 act=tf.nn.relu, placeholders=None, bias=True, featureless=False,
                 sparse_inputs=False, name='', **kwargs):
        # 初始化函数，用于初始化Dense类
        super(Dense, self).__init__(**kwargs)

        # 设置dropout率
        self.dropout = dropout
        # 设置类名
        self.name = name
        # 设置激活函数
        self.act = act
        # 设置是否为featureless
        self.featureless = featureless
        # 设置是否添加偏置
        self.bias = bias
        # 设置输入维度
        self.input_dim = input_dim
        # 设置输出维度
        self.output_dim = output_dim

        # helper variable for sparse dropout
        self.sparse_inputs = sparse_inputs
        # 如果输入是稀疏的，则将placeholders中的num_features_nonzero赋值给self.num_features_nonzero
        if sparse_inputs:
            self.num_features_nonzero = placeholders['num_features_nonzero']

        # 在变量作用域中创建变量
        with tf.variable_scope(self.name + '_vars'):
            # 创建权重变量，形状为(input_dim, output_dim)，数据类型为tf.float32，初始化器为xavier_initializer，正则化器为l2_regularizer
            self.vars['weights'] = tf.get_variable('weights', shape=(input_dim, output_dim),
                                         dtype=tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer(),
                                         regularizer=tf.contrib.layers.l2_regularizer(conf.weight_decay))
            # 如果有偏置，则创建偏置变量，形状为(output_dim)，数据类型为tf.float32，初始化为0
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

    def _call(self, inputs):
        x = inputs

        # x = tf.nn.dropout(x, self.dropout)

        # transform
        # 使用权重矩阵对输入进行线性变换
        output = tf.matmul(x, self.vars['weights'])

        # bias
        # 如果有偏置，则将偏置加到输出上
        if self.bias:
            output += self.vars['bias']

        # 激活函数
        return self.act(output)

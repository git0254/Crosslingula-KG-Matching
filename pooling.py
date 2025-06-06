import tensorflow as tf


def mean_pool(input_tensor, sequence_length=None):
    """
    Given an input tensor (e.g., the outputs of a LSTM), do mean pooling
    over the last dimension of the input.

    For example, if the input was the output of a LSTM of shape
    (batch_size, sequence length, hidden_dim), this would
    calculate a mean pooling over the last dimension (taking the padding
    into account, if provided) to output a tensor of shape
    (batch_size, hidden_dim).

    Parameters
    ----------
    input_tensor: Tensor
        An input tensor, preferably the output of a tensorflow RNN.
        The mean-pooled representation of this output will be calculated
        over the last dimension.

    sequence_length: Tensor, optional (default=None)
        A tensor of dimension (batch_size, ) indicating the length
        of the sequences before padding was applied.

    Returns
    -------
    mean_pooled_output: Tensor
        A tensor of one less dimension than the input, with the size of the
        last dimension equal to the hidden dimension state size.
    """
    with tf.name_scope("mean_pool"):
        # shape (batch_size, sequence_length)
        input_tensor_sum = tf.reduce_sum(input_tensor, axis=-2)

        # If sequence_length is None, divide by the sequence length
        # as indicated by the input tensor.
        if sequence_length is None:
            sequence_length = tf.shape(input_tensor)[-2]

        # Expand sequence length from shape (batch_size,) to
        # (batch_size, 1) for broadcasting to work.
        expanded_sequence_length = tf.cast(tf.expand_dims(sequence_length, -1),
                                           "float32") + 1e-08

        # Now, divide by the length of each sequence.
        # shape (batch_size, sequence_length)
        mean_pooled_input = (input_tensor_sum /
                             expanded_sequence_length)
        return mean_pooled_input

# 定义一个函数，用于处理填充最大池化
def handle_pad_max_pooling(tensor, last_dim):
    # 将tensor重新塑形为[-1, last_dim]的形状
    tensor = tf.reshape(tensor, [-1, last_dim])
    # 获取tensor的第一维的大小
    bs = tf.shape(tensor)[0]
    # 创建一个大小为[bs, last_dim]的全为-1e9的张量
    tt = tf.fill(tf.stack([bs, last_dim]), -1e9)
    # 判断tensor中是否等于0.0
    cond = tf.not_equal(tensor, 0.0)
    # 如果等于0.0，则将tensor中的值替换为-1e9
    res = tf.where(cond, tensor, tt)
    # 返回处理后的tensor
    return res

def max_pool(input_tensor, last_dim, sequence_length=None):
    """
    Given an input tensor, do max pooling over the last dimension of the input
    :param input_tensor:
    :param sequence_length:
    :return:
    """
    with tf.name_scope("max_pool"):
        #shape [batch_size, sequence_length]
        mid_dim = tf.shape(input_tensor)[1]
        #处理padding
        input_tensor = handle_pad_max_pooling(input_tensor, last_dim)
        #将input_tensor重新reshape为[batch_size*sequence_length, mid_dim, last_dim]
        input_tensor = tf.reshape(input_tensor, [-1, mid_dim, last_dim])
        #在mid_dim维度上取最大值
        input_tensor_max = tf.reduce_max(input_tensor, axis=-2)
        return input_tensor_max

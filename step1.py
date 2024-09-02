import numpy as np

np.random.seed(114514)


def scaled_dot_product_attention(Q, K, V, mask=None):
    '''
    1. 需要完成调整 K 的转置来匹配 Q 的最后一个维度，
    2. 计算attn_score并缩放，
    3. softmax 应用于最后一个轴计算attn_weight，
    4. 应用attn_weights输出output
    5. 带掩码mask的的注意力可以不用实现,但请记住encoder和decoder的transformer块是不一样的，很大一部分都在就在mask上
    '''
    # 1. 调整 K 的转置来匹配 Q 的最后一个维度
    K_T = K.transpose(0, 1, 3, 2)
    # 2. 计算attn_score并缩放，
    d_k = Q.shape[-1]
    attn_score = np.matmul(Q, K_T) / np.sqrt(d_k)

    # 3. softmax 应用于最后一个轴计算attn_weight
    attention_weights = np.exp(attn_score)
    attention_weights /= np.sum(attention_weights, axis=-1, keepdims=True)

    # 4. 输出output
    output = np.matmul(attention_weights, V)

    return output, attention_weights


def multi_head_attention(embed_size, num_heads, input, mask=None):
    '''
    1. embed_size 确保可以等分 num_heads 份， 否则输出错误
    2. 随机初始化Wq,Wk,Wv,Wo矩阵，并对input做线性变换
    3. 利用scaled_dot_product_attention()输出的attn_output计算O
    4. 返回output, attN_weights
    '''
    # 1. embed_size 确保可以等分 num_heads 份，否则输出错误
    assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"
    depth = embed_size // num_heads
    batch_size = input.shape[0]
    seq_len = input.shape[1]

    # 2. 随机初始化 Wq, Wk, Wv, Wo 矩阵，并对input做线性变换
    Wq = np.random.randn(embed_size, embed_size)
    Wk = np.random.randn(embed_size, embed_size)
    Wv = np.random.randn(embed_size, embed_size)
    Wo = np.random.randn(embed_size, embed_size)

    Q = np.dot(input, Wq).reshape(batch_size, seq_len, num_heads, depth).transpose(0, 2, 1, 3)
    K = np.dot(input, Wk).reshape(batch_size, seq_len, num_heads, depth).transpose(0, 2, 1, 3)
    V = np.dot(input, Wv).reshape(batch_size, seq_len, num_heads, depth).transpose(0, 2, 1, 3)

    # 3. 利用 scaled_dot_product_attention() 输出的 attn_output 计算 O
    attn_output, weights = scaled_dot_product_attention(Q, K, V, mask)

    # 连接所有头的输出
    attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, embed_size)

    # 最后的线性变换
    output = np.dot(attn_output, Wo)
    return output, weights


# test e.g.
embed_size = 128
num_heads = 8
input = np.random.randn(10, 20, embed_size)
output, weights = multi_head_attention(embed_size, num_heads, input)
print(output.shape, weights.shape)
print(output[0][0][:10], weights[0][0][0][:10])

# LLM Answer

## Step 1. Transformer

### Scaled Dot-Product Attention

$$
\text{Attention}(Q, K, V ) = \text{softmax}(\frac{QK^T} {\sqrt{d_k}} )V
$$

###  Multi-Head Attention

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$

$$
\text{where} \quad \text{head}_i = \text{Attention}(QW^Q_i, KW^K_i, VW^V_i)
$$

## Step 2. 大模型的三种架构

### 1.为什么bert不能像gpt一样做文本生成？

​	BERT 使用的是 Transformer 的编码器（encoder）部分，主要用于理解和生成上下文相关的文本表示。BERT 通过从左到右和从右到左的两个方向同时读取上下文信息。BERT 的训练任务是掩盖语言模型（MLM），这种方法要求模型预测被掩盖的单词，但这些掩盖的单词在实际生成中并不存在。BERT 通常用于自然语言理解（NLU）任务。

​	GPT 使用的是 Transformer 的解码器（decoder）部分，更适合生成任务。GPT 的训练目标是基于自回归（autoregressive）语言模型，即给定前面的文本来预测下一个单词。这样，GPT 可以逐步生成新文本，因为它能够有效地利用已经生成的上下文。

### 2.对于decoder-only的模型，它是由tokenizer，embedding层， $N\times$transformer block，lm_head，请你简单说明一下这4个部分分别在做什么？token是一个什么东西？

#### 2.1 tokenizer

Tokenizer 的作用是将输入的文本字符串转换为模型可以理解的形式（即一系列的 token）。

#### 2.2 Embedding 层

Embedding 层将 token 的索引转换为向量，这些向量表示每个 token 的语义信息。

#### 2.3 N × Transformer Block

每个 block 包括多头自注意力机制和前馈神经网络。多个 block 叠加，中间添加残差连接和层归一化，通过多个这样的 block，模型可以逐层构建更复杂的理解和表示，捕捉长距离的依赖关系。

#### 2.4  LM Head

LM Head 是模型的输出层，它将来自最后一个 Transformer block 的输出向量转换回 token 的预测分布。

#### 2.5 Token 是什么:

Token 是指文本中被分割后的基本单元，通常是一个词或子词。在自然语言处理中，token 是模型处理的基本对象。Token 通过词汇表中的索引表示，用于表示语言的结构和内容。

### 3.为什么decoder-only的模型的数量远远多于Encoder-Decoder模型？

1.Decoder-only 模型的架构相对更简单，只使用了 Transformer 的解码器部分。由于只需要单向生成，训练过程更为高效。Decoder-only 模型通常更快，并且对计算资源的需求较低，易于扩展到更大规模。

2.Decoder-only 模型的预训练过程相对简单，只需要大规模无监督文本数据，进行自回归语言建模，可通过不断增加参数和数据量，具有推广性和可扩展性。

### 4.使用预训练好的bert/gpt以及它们对应的tokenizer在imdb任务上finetune，计算这两种模型在IMDB分类任务在测试集上的准确率，并比较二者在训练前后分类的准确性




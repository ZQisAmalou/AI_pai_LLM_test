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

### 2.1 为什么bert不能像gpt一样做文本生成？

​	BERT 使用的是 Transformer 的 encoder 部分，主要用于理解和生成上下文相关的文本表示。BERT 通过从左到右和从右到左的两个方向同时读取上下文信息。BERT 的训练任务是预测被掩盖的单词，但这些掩盖的单词在实际生成中并不存在，因此BERT 通常用于自然语言理解任务，而不能做文本生成任务。

​	GPT 使用的是 Transformer 的 decoder 部分，更适合生成任务。GPT 的训练目标是给定前面的文本来预测下一个单词。因此GPT 可以能够有效地利用已经生成的上下文，逐步生成新文本。

### 2.2 对于decoder-only的模型，它是由tokenizer，embedding层， $N\times$transformer block，lm_head，请你简单说明一下这4个部分分别在做什么？token是一个什么东西？

#### 2.2.1 tokenizer

Tokenizer 的作用是将输入的文本字符串转换为模型可以理解的形式（即一系列的 token）。

#### 2.2.2 Embedding 层

Embedding 层将 token 的索引转换为向量，表示每个 token 的语义信息。

#### 2.2.3 N × Transformer Block

每个 block 包括多头自注意力机制和前馈神经网络。多个 block 叠加，中间添加残差连接和层归一化，通过多个这样的 block，模型可以逐层构建更复杂的理解和表示，从而捕捉长距离的依赖关系。

#### 2.2.4  LM Head

LM Head 根据 layer 层输出的 logits 通过 softmax 计算概率，若在推理中，则根据概率通过解码算法得到 next token；若在训练中，则根据概率和标签计算损失，并开始反向过程。

#### 2.2.5 Token 是什么

Token 是指文本中被分割后的基本单元，通常是一个词或子词。在自然语言处理中，token 是模型处理的基本对象。Token 通过词汇表中的索引表示，用于表示语言的结构和内容。

### 2.3 为什么decoder-only的模型的数量远远多于Encoder-Decoder模型？

1.Decoder-only 模型的架构相对更简单，只使用了 Transformer 的解码器部分。由于只需要单向生成，训练过程更为高效。Decoder-only 模型通常更快，并且对计算资源的需求较低，易于扩展到更大规模。

2.Decoder-only 模型的预训练过程相对简单，只需要大规模无监督文本数据，进行自回归语言建模，可通过不断增加参数和数据量，具有推广性和可扩展性。

### 2.4 使用预训练好的bert/gpt以及它们对应的tokenizer在imdb任务上finetune，计算这两种模型在IMDB分类任务在测试集上的准确率，并比较二者在训练前后分类的准确性

预训练好的准确率

![Figure_H1](images\BERT.png)

![Figure_H1](images\GPT2.png)

finetune后的准确率

![Figure_H1](images\BERT_trained.png)

![Figure_H1](images\GPT2_trained.png)

训练前后两模型均取得了较大提升。

## Step 3. 一个decoder-only的Generative LLM的前世今生

### 3.1 Pre-training

#### 	max_length是做什么的？

​	max_length 参数用于限制模型处理的输入序列的最大长度。具体来说，它决定了模型在每次前向传播中所能处理的最大文本长度。如果输入的文本序列长度超过了这个值，模型会进行截断。

#### 	warm up 的作用

​	训练开始时的 warm-up 阶段指的是学习率在初始阶段逐步增加的过程。通过设定一个递增的步数，学习率从较小的值开始逐渐增大。这种做法的目的是避免模型在刚开始训练时由于权重尚未优化而导致梯度更新过大，进而引起训练的不稳定甚至发散。通过 warm-up，可以让模型在训练初期更加平稳，从而提高整体训练效果。

### 3.2 Post-training

#### 3.2.1 Instruction Tuning

##### 	什么是instruction tuning？为什么需要instruction tuning?

​	Instruction Tuning 是对预训练的大型语言模型进行微调，使其能够更好地理解和执行人类指令。这一过程通常学习大量高质量的指令响应数据来提高在人类指令下完成任务的能力。

​	需要 Instruction Tuning 的原因：预训练模型原始训练目标并不关注在人类指令下完成任务，生成文本。通过指令微调，可以让模型更好地理解复杂的人类指令，从而更好的完成任务。

##### 	Llama3 的instruction tuning的格式是怎么样的？

```
<s>[INST] <<SYS>>
{system_message}
<</SYS>>
{user_message} [/INST]
{assistant_message}
```

#### 3.2.2 SFT

##### 	为什么要将llm与人类偏好对齐？不这么做会出现什么问题？

​	确保模型生成的内容符合人类期望，并能够更好地在真实场景中与用户进行有效的互动。

​	不这么做可能会出现以下问题：

​	**1.生成不符合用户期望的回答**：
​		预训练的语言模型虽然在大量数据上进行了训练，具备丰富的知识，但并不能完全理解用户的意图或偏好，或是不能理解人类的	指令。如果不进行人类偏好的对齐，模型可能会生成难以在实际应用中提供有价值的、合乎情境的回应。

​	**2.产生不安全或有害的内容**：
​		预训练模型可能会生成不适合的或有害的内容，比如带有偏见、不安全的建议或有争议的言论。

#### 3.2.3 RLHF/PPO

##### 	RLHF 的偏好数据集是如何构造的？

​	**1.模型生成多个回答**：对于每个用户的输入指令，模型生成多个不同的回答版本。

​	**2.人类评估**：人类标注员根据回答的质量进行评估，选择他们认为最好的回答，并标记为优选的回答。

​	**3.偏好对比**：通过人类的偏好选择，构建出偏好对比样本，并用这些偏好数据来训练奖励模型（reward model）。

##### 	Reward Model是做什么的？它是如何被训练的？

​	Reward Model 是用来评估模型生成回答的质量的工具。根据人类标注的偏好数据来计算每个回答的得分，从而帮助强化学习算法选择最优的回答。

**训练过程**：

​	**1.输入数据**：输入人类偏好数据集，包括模型生成的多个回答以及人类对这些回答的偏好排序。

​	**2.训练目标**：目标是学会为每个回答生成一个得分，并确保这些得分与人类的偏好一致。

​	**3.损失函数**：通常使用对比损失（如贝叶斯优化或对比学习损失），最大化人类选择的偏好回答的分数。

#### 3.2.4 PPO

##### 	DPO和PPO相比优势在哪里？

​	1.更低的实现复杂度和更高的稳定性

​	PPO 依赖于策略梯度方法，通过采样并计算 reward 来更新策略。训练过程需对模型进行采样、回报估计、策略更新等多步操作，且强化学习训练容易出现不稳定性，需要仔细调整超参数（如回报折扣因子、探索和利用的平衡等）来保证模型的收敛性。

​	DPO 不需要构建复杂的强化学习框架，其主要思想是直接优化模型，使其生成符合人类偏好的答案。模型训练只需要处理偏好数据，而不依赖于复杂的回报估计，因此训练更加稳定，减少了不稳定因素。

​	2.效率更高，计算资源消耗更小，用时更短

​	PPO 由于要处理强化学习的回报计算和策略更新，通常涉及大量的采样和探索过程，导致训练时间较长，计算资源消耗较大。

​	DPO 不需要强化学习中的探索过程，完全基于人类偏好数据进行优化。DPO 优化生成结果的偏好排名，避免强化学习中的复杂回报估计和采样步骤，从而大幅减少了计算资源的使用，且训练时间比更短。

## Step 4. LLM 实战演练1

​	Gemma-2-2b-it的测试结果

​	![Figure_H1](images\step4.png)

## Step 5. LLM 实战演练2

​	glm-4-flash api 的测试结果

![Figure_H1](images\step5.png)

## Step 6. Research 

### 6.1 复现CodeAttack

#### 文章发现：

1.codeattack和自然语言间的分布差距较大导致了安全泛化能力弱

2.更强大的模型并不一定更安全

3.不同的编程语言在安全行为上表现不一致，即编程语言的不平衡分布拉大了安全泛化性差距。

#### CodeAttack 框架

1.输入编码

2.任务理解

3.输出规范

#### 复现结果

用glm-4-flash复现代码生成攻击，进行的SQL注入漏洞、文件读取漏洞（路径遍历攻击）、使用不安全的哈希算法等攻击均得到了大模型的安全性修正，都通过了静态安全分析工具 bandit 的检查。

结果：

```
Run metrics:
	Total issues (by severity):
		Undefined: 0
		Low: 0
		Medium: 0
		High: 0
	Total issues (by confidence):
		Undefined: 0
		Low: 0
		Medium: 0
		High: 0
```

### 6.2 关于越狱攻击的防御

##### 1.研究的问题

​	目前研究的安全性防御虽然在方法和模型上取得了进步，但安全调优数据的影响没有得到充分探索。作者在安全调优数据中发现了拒绝位置偏差，阻碍了调优后的LLM学习如何有效地拒绝的能力。在生成响应之前做出决策导致的缺点：1.缺乏做出拒绝决策需要的信息；2.没有在响应后期阶段纳入拒绝的机制。

##### 2.解决方案

##### Decoupled Refusal Training(DeRTa):

​	MLE with Harmful Response Prefix：在安全响应前附加一段随机长度的有害响应

​	Reinforced Transition Optimization (RTO)：针对有害前缀策略，引入一个辅助训练目标，使模型能够在整个有害响应序列的任意位置实现从有害到安全的平滑过渡，并做出拒绝决策。


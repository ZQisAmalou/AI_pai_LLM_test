import os
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score

# 使用镜像站加速下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 加载IMDB数据集
ds = load_dataset("stanfordnlp/imdb")

# 定义评价指标
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

# BERT模型部分

# 加载BERT模型和tokenizer
model_name_bert = "google-bert/bert-base-uncased"
tokenizer_bert = AutoTokenizer.from_pretrained(model_name_bert)
model_bert = AutoModelForSequenceClassification.from_pretrained(model_name_bert, num_labels=2)

# 数据预处理函数
def preprocess_function_bert(examples):
    return tokenizer_bert(examples['text'], truncation=True)

# 应用数据预处理
encoded_ds_bert = ds.map(preprocess_function_bert, batched=True)

# 训练参数设置
training_args_bert = TrainingArguments(
    output_dir="./results_bert",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir="./logs_bert",
    logging_steps=10,
    load_best_model_at_end=True
)

# 创建Trainer并训练BERT
trainer_bert = Trainer(
    model=model_bert,
    args=training_args_bert,
    train_dataset=encoded_ds_bert["train"],
    eval_dataset=encoded_ds_bert["test"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer_bert,
    data_collator=DataCollatorWithPadding(tokenizer_bert)
)

# 开始训练和评估
trainer_bert.train()
results_bert = trainer_bert.evaluate()
print(f"BERT accuracy: {results_bert['eval_accuracy']:.4f}")

# GPT-2模型部分

# 加载GPT-2模型和tokenizer
model_name_gpt2 = "openai-community/gpt2"
tokenizer_gpt2 = AutoTokenizer.from_pretrained(model_name_gpt2)
model_gpt2 = AutoModelForSequenceClassification.from_pretrained(model_name_gpt2, num_labels=2)

# 由于GPT-2没有默认的填充token，需要指定一个填充token
tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token

# 数据预处理函数
def preprocess_function_gpt2(examples):
    return tokenizer_gpt2(examples['text'], truncation=True)

# 应用数据预处理
encoded_ds_gpt2 = ds.map(preprocess_function_gpt2, batched=True)

# 训练参数设置
training_args_gpt2 = TrainingArguments(
    output_dir="./results_gpt2",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir="./logs_gpt2",
    logging_steps=10,
    load_best_model_at_end=True
)

# 创建Trainer并训练GPT-2
trainer_gpt2 = Trainer(
    model=model_gpt2,
    args=training_args_gpt2,
    train_dataset=encoded_ds_gpt2["train"],
    eval_dataset=encoded_ds_gpt2["test"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer_gpt2,
    data_collator=DataCollatorWithPadding(tokenizer_gpt2)
)

# 开始训练和评估
trainer_gpt2.train()
results_gpt2 = trainer_gpt2.evaluate()
print(f"GPT-2 accuracy: {results_gpt2['eval_accuracy']:.4f}")

import os, sys

class ModelConf(object):
    """All Self-Attention params"""

    def __init__(self):
        self.embedding_dim = 256  # 输入向量维度
        self.seq_length = 30  # 序列长度
        self.lstm_hidden = 512  # rnn 隐层维度
        self.attention_hidden = 256  # attention 隐层维度
        self.attention_num = 32  # attention 观点数
        self.penalty_C = 0.0  # attention矩阵二范数惩罚系数
        self.fc1_dim = 1024  # 全连接维度(第1层)
        self.output_dim = 256  # 模块encode输出向量维度

    def __str__(self):
        output = "embedding_dim: " + str(self.embedding_dim) + "\n" + \
                 "seq_length: " + str(self.seq_length) + "\n" + \
                 "lstm_hidden: " + str(self.lstm_hidden) + "\n" + \
                 "attention_hidden: " + str(self.attention_hidden) + "\n" + \
                 "attention_num: " + str(self.attention_num) + "\n" + \
                 "penalty_C: " + str(self.penalty_C) + "\n" + \
                 "fc1_dim: " + str(self.fc1_dim) + "\n" + \
                 "output_dim: " + str(self.output_dim) + "\n"
        return output




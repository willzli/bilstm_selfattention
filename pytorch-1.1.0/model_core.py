import os, sys

import torch
from .model_conf import ModelConf

class ModelCore(torch.nn.Module):
    
    def __init__(self):
        super(ModelCore, self).__init__()

        # load conf and dict
        self.model_conf = ModelConf()

        # define model var
        if self.model_conf.rnn_type == "lstm":
            self.rnn_model = torch.nn.LSTM(input_size=self.model_conf.embedding_dim, hidden_size=self.model_conf.rnn_hidden, num_layers=1, batch_first=True, bidirectional=True)
            self.attention_input_dim = 2*self.model_conf.rnn_hidden
        elif self.model_conf.rnn_type == "gru":
            self.rnn_model = torch.nn.GRU(input_size=self.model_conf.embedding_dim, hidden_size=self.model_conf.rnn_hidden, num_layers=1, batch_first=True, bidirectional=True)
            self.attention_input_dim = 2*self.model_conf.rnn_hidden
        elif self.model_conf.rnn_type == "no":
            self.rnn_model = None
            self.attention_input_dim = self.model_conf.embedding_dim
        else:
            print("rnn_type error !")
            sys.exit(1)
        self.linear_W_s1 = torch.nn.Linear(in_features=self.attention_input_dim, out_features=self.model_conf.attention_hidden)
        self.linear_W_s2 = torch.nn.Linear(in_features=self.model_conf.attention_hidden, out_features=self.model_conf.attention_num)
        self.linear_fc = torch.nn.Linear(in_features=self.attention_input_dim*self.model_conf.attention_num, out_features=self.model_conf.fc1_dim)
        self.linear_output = torch.nn.Linear(in_features=self.model_conf.fc1_dim, out_features=self.model_conf.output_dim)
        
        self.dropout = torch.nn.Dropout(p=(1.0-self.model_conf.dropout_keep_prob))

        I = torch.eye(self.model_conf.attention_num)
        self.register_buffer("I", I)


    def forward(self, embedding_input):
        
        # selfattention encode
        if self.model_conf.rnn_type != "no":
            birnn_output, _ = self.rnn_model(embedding_input)
        else:
            birnn_output = embedding_input
        H_s1 = torch.tanh(self.linear_W_s1(birnn_output))
        H_s2 = self.linear_W_s2(H_s1)
        A = torch.nn.functional.softmax(input=H_s2, dim=1)
        A_trans = A.transpose(1, 2)
        M = torch.matmul(A_trans, birnn_output)
        M_flat = torch.reshape(input=M, shape=[-1, self.attention_input_dim * self.model_conf.attention_num])
        fc = self.linear_fc(M_flat)
        fc = torch.nn.functional.leaky_relu(input=fc, negative_slope=self.model_conf.relu_leakiness)
        if self.model_conf.use_dropout == True:
            fc = self.dropout(fc)
        fc = self.linear_output(fc)
        
        # selfattention penalization
        I = self.I.repeat([embedding_input.shape[0], 1, 1])
        AA_T = torch.matmul(A_trans, A)
        P = torch.pow(torch.norm(input=(AA_T - I), p='fro', dim=[-2, -1]), 2.0)
        loss_P = self.model_conf.penalty_C * torch.mean(P)

        return fc, loss_P
    

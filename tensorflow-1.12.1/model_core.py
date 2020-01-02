import os, sys
import tensorflow as tf
from .model_conf import ModelConf

class ModelCore(object):

    def __init__(self):
        self.model_conf = ModelConf()

    def encode(self, input_x):

        with tf.variable_scope("embedding"):
            embedding_inputs = input_x
        
        with tf.variable_scope("lstm"):
            lstm_fw = tf.nn.rnn_cell.LSTMCell(num_units=self.model_conf.lstm_hidden, name='lstm_fw')
            lstm_bw = tf.nn.rnn_cell.LSTMCell(num_units=self.model_conf.lstm_hidden, name='lstm_bw')
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw, cell_bw=lstm_bw, inputs=embedding_inputs, dtype=tf.float32)
            lstm_output = tf.concat([output_fw, output_bw], axis=2)
            lstm_output_rsp = tf.reshape(lstm_output, [-1, 2 * self.model_conf.lstm_hidden])

        with tf.variable_scope("self-attention"):
            W_s1 = tf.get_variable(name='W_s1', shape=[2*self.model_conf.lstm_hidden, self.model_conf.attention_hidden])
            H_s1 = tf.nn.tanh(tf.matmul(lstm_output_rsp, W_s1))
            W_s2 = tf.get_variable(name='W_s2', shape=[self.model_conf.attention_hidden, self.model_conf.attention_num])
            H_s2 = tf.matmul(H_s1, W_s2)
            H_s2_rsp = tf.transpose(tf.reshape(H_s2, [-1, self.model_conf.seq_length, self.model_conf.attention_num]), [0, 2, 1])
            A = tf.nn.softmax(logits=H_s2_rsp, axis=-1, name="attention")
            self.heat_matrix = A
            M = tf.matmul(A, lstm_output)
            M_flat = tf.reshape(M, [-1, 2 * self.model_conf.lstm_hidden * self.model_conf.attention_num])

        with tf.variable_scope("penalization"):
            AA_T = tf.matmul(A, tf.transpose(A, [0, 2, 1]))
            I = tf.eye(self.model_conf.attention_num, batch_shape=[tf.shape(A)[0]])
            P = tf.square(tf.norm(AA_T - I, axis=[-2, -1], ord="fro"))
            loss_P = tf.reduce_mean(self.model_conf.penalty_C * P)

        with tf.variable_scope("dense"):
            fc = tf.layers.dense(inputs=M_flat, units=self.model_conf.fc1_dim, activation=None, name='fc1')
            fc = tf.nn.relu(fc)
            fc = tf.layers.dense(inputs=fc, units=self.model_conf.output_dim, activation=None, name='fc_out')

        return fc, loss_P






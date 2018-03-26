import tensorflow as tf  
import numpy as np  
import matplotlib.pyplot as plt  
import os
import random
  
# 导入MNIST数据  
#from tensorflow.examples.tutorials.mnist import input_data  
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  

# 导入波形数据
from fileIO import readfile, dimstandard, next_batch
aimFile = "VIBMON_K1702B_1H.txt"
origin = readfile(aimFile)[:200]
data_all = dimstandard(origin)
random.shuffle(data_all)
data = data_all
# print(data)
# exit()


class endecode():
    def __init__(self): 
        self.learning_rate = 0.0001
        self.training_epochs = 10000  
        self.display_step = 5  
        self.examples_to_show = 10  
        self.n_input = 4000 
        self.batch_size = 25
        self.all_batch = len(data)
      
        # tf Graph input (only pictures)  
        self.X = tf.placeholder("float", [None, self.n_input]) 
        self.n_hidden_1 = 1024
        self.n_hidden_2 = 16 # 第一编码层神经元个数  
#        self.n_hidden_3 = 32 # 第二编码层神经元个数  
#        self.n_hidden_4 = 64  
        # self.n_hidden_5 = 64 
        # 权重和偏置的变化在编码层和解码层顺序是相逆的  
        # 权重参数矩阵维度是每层的 输入*输出，偏置参数维度取决于输出层的单元数  
        self.W_encoder_h1 = tf.Variable(tf.truncated_normal(shape = [self.n_input,    self.n_hidden_1], stddev=0.1), name = 'W_encoder_h1')  
        self.W_encoder_h2 = tf.Variable(tf.truncated_normal(shape = [self.n_hidden_1, self.n_hidden_2], stddev=0.1), name = 'W_encoder_h2')  
#        self.W_encoder_h3 = tf.Variable(tf.truncated_normal(shape = [self.n_hidden_2, self.n_hidden_3], stddev=0.1), name = 'W_encoder_h3') 
#        self.W_encoder_h4 = tf.Variable(tf.truncated_normal(shape = [self.n_hidden_3, self.n_hidden_4], stddev=0.1), name = 'W_encoder_h4')  
        self.W_decoder_h1 = tf.Variable(tf.truncated_normal(shape = [self.n_hidden_2, self.n_hidden_1], stddev=0.1), name = 'W_decoder_h1')  
        self.W_decoder_h2 = tf.Variable(tf.truncated_normal(shape = [self.n_hidden_1, self.n_input], stddev=0.1), name = 'W_decoder_h2')  
#        self.W_decoder_h3 = tf.Variable(tf.truncated_normal(shape = [self.n_hidden_1, self.n_input], stddev=0.1), name = 'W_decoder_h3')  
#        self.W_decoder_h4 = tf.Variable(tf.truncated_normal(shape = [self.n_hidden_1, self.n_input], stddev=0.1), name = 'W_decoder_h4')
        # self.W_decoder_h5 = tf.Variable(tf.truncated_normal(shape = [self.n_hidden_1, self.n_input   ], stddev=0.1), name = 'W_decoder_h5')

        self.B_encoder_b1 = tf.Variable(tf.random_normal(shape = [self.n_hidden_1]), name = 'B_encoder_b1')
        self.B_encoder_b2 = tf.Variable(tf.random_normal(shape = [self.n_hidden_2]), name = 'B_encoder_b2')
#        self.B_encoder_b3 = tf.Variable(tf.random_normal(shape = [self.n_hidden_3]), name = 'B_encoder_b3')
#        self.B_encoder_b4 = tf.Variable(tf.random_normal(shape = [self.n_hidden_4]), name = 'B_encoder_b4') 
        # self.B_encoder_b5 = tf.Variable(tf.random_normal(shape = [self.n_hidden_5]), name = 'B_encoder_b5')
        self.B_decoder_b1 = tf.Variable(tf.random_normal(shape = [self.n_hidden_1]), name = 'B_decoder_b1') 
        self.B_decoder_b2 = tf.Variable(tf.random_normal(shape = [self.n_input]), name = 'B_decoder_b2')  
#        self.B_decoder_b3 = tf.Variable(tf.random_normal(shape = [self.n_input]), name = 'B_decoder_b3') 
#        self.B_decoder_b4 = tf.Variable(tf.random_normal(shape = [self.n_input]), name = 'B_decoder_b4')  
        # self.B_decoder_b5 = tf.Variable(tf.random_normal(shape = [self.n_input]), name = 'B_decoder_b5')  

    # 每一层结构都是 xW + b  
    # 构建编码器  
    def encoder(self, x):  
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.W_encoder_h1), self.B_encoder_b1))  
        layer_2 = tf.add(tf.matmul(layer_1, self.W_encoder_h2), self.B_encoder_b2) 
#        layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, self.W_encoder_h3), self.B_encoder_b3)) 
#        layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, self.W_encoder_h4), self.B_encoder_b4))
        return layer_2  
      
    # 构建解码器  
    def decoder(self, x):  
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.W_decoder_h1), self.B_decoder_b1))  
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.W_decoder_h2), self.B_decoder_b2)) 
        # layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.W_encoder_h2), self.B_encoder_b2))   
#        layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, self.W_decoder_h3), self.B_decoder_b3))  
#        layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, self.W_decoder_h4), self.B_decoder_b4))  
        # layer_5 = tf.nn.sigmoid(tf.add(tf.matmul(layer_4, self.W_decoder_h5), self.B_decoder_b5)) 
        return layer_2
    
    def train(self):
        # 构建模型  
        encoder_op = self.encoder(self.X)  
        decoder_op = self.decoder(encoder_op)  
      
        # 预测  
        y_pred = decoder_op  
        y_true = self.X  
          
        # 定义代价函数和优化器  
        cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2)) #最小二乘法  
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)  
          
        with tf.Session() as sess:  
            saver = tf.train.Saver(max_to_keep = 1)
            '''
            加载保存的参数
            '''
            if os.path.exists("./param/checkpoint"):
                saver.restore(sess, "./param/selfencoding.ckpt")
                print('hhh')
#                print("W_encoder_h1:", sess.run(self.W_encoder_h1))  
            else:      
                sess.run(tf.global_variables_initializer())
                print('heiheihei')
            
            # 首先计算总批数，保证每次循环训练集中的每个样本都参与训练 
            total_batch = int(self.all_batch/self.batch_size) #总批数  
            min_cost = 1
            for epoch in range(1, self.training_epochs + 1):  
                count = 0
                for i in range(self.all_batch + 1):                    
                    if i > 0 and i % self.batch_size == 0:
                        batch_xs = data[count: i]
                        _, acc = sess.run([optimizer, cost], feed_dict={self.X: batch_xs})  
                        count = i
                        # print(count)
                if epoch % self.display_step == 0:  
                    print("Epoch:", '%04d' % (epoch), "cost=", "{:.9f}".format(acc))   
                if acc < min_cost:
                    min_cost = acc
                    saver.save(sess, "./param/selfencoding.ckpt")
            print("Training Finished!")  

if __name__ == '__main__':    
    a = endecode()
    a.train()
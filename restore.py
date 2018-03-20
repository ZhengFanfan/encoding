import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from fileIO import readfile, dimstandard, normalization, convert
parentPath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
aimPath = parentPath + '//features//DLSH//K1702B_3//normal'
aimFile = os.path.join(aimPath, "VIBMON_K1702B_1H.txt")

#exit()
#plt.title('oringinal')
#data = np.array(data)
# print(data)

n_input = 4000
X = tf.placeholder("float", [None, n_input])
n_hidden_1 = 1024  # 第一编码层神经元个数
n_hidden_2 = 8  # 第二编码层神经元个数
#n_hidden_3 = 256
#n_hidden_4 = 64 
# 权重和偏置的变化在编码层和解码层顺序是相逆的
# 权重参数矩阵维度是每层的 输入*输出，偏置参数维度取决于输出层的单元数
W_encoder_h1 = tf.Variable(tf.truncated_normal(shape = [n_input,    n_hidden_1], stddev=0.1), name = 'W_encoder_h1')  
W_encoder_h2 = tf.Variable(tf.truncated_normal(shape = [n_hidden_1, n_hidden_2], stddev=0.1), name = 'W_encoder_h2')  
#W_encoder_h3 = tf.Variable(tf.truncated_normal(shape = [n_hidden_2, n_hidden_3], stddev=0.1), name = 'W_encoder_h3') 
#W_encoder_h4 = tf.Variable(tf.truncated_normal(shape = [n_hidden_3, n_hidden_4], stddev=0.1), name = 'W_encoder_h4')  
W_decoder_h1 = tf.Variable(tf.truncated_normal(shape = [n_hidden_2, n_hidden_1], stddev=0.1), name = 'W_decoder_h1')  
W_decoder_h2 = tf.Variable(tf.truncated_normal(shape = [n_hidden_1, n_input], stddev=0.1), name = 'W_decoder_h2')  
#W_decoder_h3 = tf.Variable(tf.truncated_normal(shape = [n_hidden_2, n_hidden_1], stddev=0.1), name = 'W_decoder_h3')  
#W_decoder_h4 = tf.Variable(tf.truncated_normal(shape = [n_hidden_1, n_input   ], stddev=0.1), name = 'W_decoder_h4')
 
B_encoder_b1 = tf.Variable(tf.random_normal(shape = [n_hidden_1]), name = 'B_encoder_b1')
B_encoder_b2 = tf.Variable(tf.random_normal(shape = [n_hidden_2]), name = 'B_encoder_b2')
#B_encoder_b3 = tf.Variable(tf.random_normal(shape = [n_hidden_3]), name = 'B_encoder_b3')
#B_encoder_b4 = tf.Variable(tf.random_normal(shape = [n_hidden_4]), name = 'B_encoder_b4') 
B_decoder_b1 = tf.Variable(tf.random_normal(shape = [n_hidden_1]), name = 'B_decoder_b1') 
B_decoder_b2 = tf.Variable(tf.random_normal(shape = [n_input]), name = 'B_decoder_b2')  
#B_decoder_b3 = tf.Variable(tf.random_normal(shape = [n_hidden_1]), name = 'B_decoder_b3') 
#B_decoder_b4 = tf.Variable(tf.random_normal(shape = [n_input]), name = 'B_decoder_b4')  


def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, W_encoder_h1), B_encoder_b1))  
    layer_2 = tf.add(tf.matmul(layer_1, W_encoder_h2), B_encoder_b2)  
#    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, W_encoder_h3), B_encoder_b3)) 
#    # 为了便于编码层的输出，编码层随后一层不使用激活函数  
#    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, W_encoder_h4), B_encoder_b4))
    return layer_2 


# 构建解码器
def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, W_decoder_h1), B_decoder_b1))  
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, W_decoder_h2), B_decoder_b2))  
#    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, W_decoder_h3), B_decoder_b3))  
#    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, W_decoder_h4), B_decoder_b4))  
    return layer_2  


# 构建模型
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "./param/selfencoding.ckpt")      
    for num in range(200, 215):
        data = readfile('./VIBMON_K1702B_1H.txt')[num]
        data = convert(data)
        data = normalization(data)

        plt.figure(1)
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)
        ax1.cla()
        ax1.plot(data, linewidth = 0.5, label = 'original')
        
        try:

            encoder_result = sess.run(encoder_op, feed_dict={X: [data]})
            result_encode = encoder_result[0]
            encode_decode = sess.run(decoder_op, feed_dict={X: [data]})  
            result_decode = encode_decode[0]
            ax2.cla()
            ax2.plot(result_decode, label = 'decoding', linewidth = 0.5, color = 'red')
            # plt.show()
            plt.savefig('vision/%s' % (num))  
        except Exception as e:
            raise e
        
    



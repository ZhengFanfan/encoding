import tensorflow as tf
import matplotlib.pyplot as plt
from fileIO import readfile, dimstandard

n_input = 4000
X = tf.placeholder("float", [None, n_input])
n_hidden_1 = 1024  # 第一编码层神经元个数
n_hidden_2 = 16  # 第二编码层神经元个数
# 权重和偏置的变化在编码层和解码层顺序是相逆的
# 权重参数矩阵维度是每层的 输入*输出，偏置参数维度取决于输出层的单元数
W_encoder_h1 = tf.Variable(tf.truncated_normal(shape = [n_input,    n_hidden_1], stddev=0.1), name = 'W_encoder_h1')  
W_encoder_h2 = tf.Variable(tf.truncated_normal(shape = [n_hidden_1, n_hidden_2], stddev=0.1), name = 'W_encoder_h2')  
W_decoder_h1 = tf.Variable(tf.truncated_normal(shape = [n_hidden_2, n_hidden_1], stddev=0.1), name = 'W_decoder_h1')  
W_decoder_h2 = tf.Variable(tf.truncated_normal(shape = [n_hidden_1, n_input], stddev=0.1), name = 'W_decoder_h2')  
 
B_encoder_b1 = tf.Variable(tf.random_normal(shape = [n_hidden_1]), name = 'B_encoder_b1')
B_encoder_b2 = tf.Variable(tf.random_normal(shape = [n_hidden_2]), name = 'B_encoder_b2')
B_decoder_b1 = tf.Variable(tf.random_normal(shape = [n_hidden_1]), name = 'B_decoder_b1') 
B_decoder_b2 = tf.Variable(tf.random_normal(shape = [n_input]), name = 'B_decoder_b2')  


def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, W_encoder_h1), B_encoder_b1))  
    layer_2 = tf.add(tf.matmul(layer_1, W_encoder_h2), B_encoder_b2)  
    return layer_2 


# 构建解码器
def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, W_decoder_h1), B_decoder_b1))  
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, W_decoder_h2), B_decoder_b2))  
    return layer_2  

def restore(data):
    # 构建模型
    encoder_op = encoder(X)
    decoder_op = decoder(encoder_op)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        
        saver.restore(sess, "./param/selfencoding.ckpt")   
        plt.figure(1)
        for num in range(0, len(data)):                  
            plt.figure(1)
            ax1 = plt.subplot(611)
            ax2 = plt.subplot(613)
            ax3 = plt.subplot(615)
            ax1.cla()
            ax1.plot(data[num], linewidth = 0.5, label = 'original')
            ax1.set_ylim([0, 1])
            try:
                encode_result = sess.run(encoder_op, feed_dict={X: [data[num]]})
                result_encode = encode_result[0]
                encode_decode = sess.run(decoder_op, feed_dict={X: [data[num]]})  
                result_decode = encode_decode[0]
                ax2.cla()
                ax2.plot(result_encode, label = 'encoding', linewidth = 0.5, color = 'green')
                ax3.cla()
                ax3.plot(result_decode, label = 'decoding', linewidth = 0.5, color = 'red')
                ax3.set_ylim([0, 1])                
                plt.savefig('vision/%s' % (num + 200))                  
            except Exception as e:
                raise e
                
if __name__ == '__main__':             
    data = readfile('./VIBMON_K1702B_1H.txt')[200:]
    data = dimstandard(data)     
    restore(data)
#    encoder_op = encoder(X)
#    decoder_op = decoder(encoder_op)
#    saver = tf.train.Saver()
#    with tf.Session() as sess:
#        plt.figure(1)
#        saver.restore(sess, "./param/selfencoding.ckpt")  
#        for num in range(0, len(data)):  
#            encode_result = sess.run(encoder_op, feed_dict={X: [data[num]]})
#            result_encode = encode_result[0]
#            plt.plot(result_encode, label = 'encoding', linewidth = 0.5, color = 'green')
##            plt.ylim((0, 1))  
#        plt.savefig('vision/%s' % ('all_feature'))  
   



#import mnist data and modules
import tensorflow as tf
import sys
sys.path.append("/mnist_data")
import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import numpy as np
def xavier_init(n_input,n_output,uniform=True):
    if uniform:
        init_range=tf.sqrt(6.0/(n_input+n_output))
        return tf.random_uniform_initializer(-init_range,init_range)
    else:
        stddev=tf.sqrt(3.0/(n_input+n_output))
        return tf.truncated_normal_initializer(stddev=stddev)
#input data
x=tf.placeholder("float",[None,784]) 
y=tf.placeholder("float",[None,10])
k=5
theta0=tf.get_variable("theta0",shape=[784,256],initializer=xavier_init(784,256))
theta=[]
for i in range(k-2):
    theta.append(tf.get_variable("theta"+str(i+2),shape=[256,256],initializer=xavier_init(256,256)))
Theta=tf.get_variable("theta_last",shape=[256,10],initializer=xavier_init(256,10))
b=tf.Variable(tf.random_normal([k-1,256]))
B=tf.Variable(tf.random_normal([10]))
layer=[]
for i in range(k):
    if i==0:
        layer.append(tf.nn.softmax(tf.matmul(x,theta0)+b[0]))
    elif i==k-1:
        Layer=tf.matmul(layer[i-1],Theta)+B
    else:
        layer.append(tf.nn.relu(tf.matmul(layer[i-1],theta[i-1])+b[i]))
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Layer,y))
#black box
a=tf.Variable(0.0003)
opt=tf.train.AdamOptimizer(a)
train=opt.minimize(cost)
#initialize
sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)
#train
for i in range(2000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train, feed_dict={x: batch_xs, y: batch_ys})
  print(sess.run(cost,feed_dict={x: batch_xs, y: batch_ys}))
  
#accuracy check
correct_prediction = tf.equal(tf.argmax(Layer,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
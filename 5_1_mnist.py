#import mnist data and modules
import tensorflow as tf
import sys
sys.path.append("/mnist_data")
import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import numpy as np
#input data
x=tf.placeholder("float") #x=nX784
y=tf.placeholder("float") #y=1X10
theta=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10])) #1X1 mat
Layer=tf.nn.softmax(tf.matmul(x,theta)+b)
cost=tf.reduce_mean(-tf.reduce_sum(y*tf.log(Layer),reduction_indices=1))
#black box
a=tf.Variable(0.1)
opt=tf.train.GradientDescentOptimizer(a)
train=opt.minimize(cost)
#initialize
sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)
#train
for i in range(3000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train, feed_dict={x: batch_xs, y: batch_ys})
  print(sess.run(cost,feed_dict={x: batch_xs, y: batch_ys}))
  
#accuracy check
correct_prediction = tf.equal(tf.argmax(Layer,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
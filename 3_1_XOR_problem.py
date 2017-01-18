import tensorflow as tf
#input
x_data=[[1.,1.],[0.,0.],[0.,1.],[1.,0.]] #4X2 mat
y_data=[[0.],[0.],[1.],[1.]]         #4X1 mat
x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)
theta1=tf.Variable(tf.random_uniform([len(x_data[0]),len(y_data[0])+1],-1.0,1.0)) #2X2 mat
theta2=tf.Variable(tf.random_uniform([len(y_data[0])+1,len(y_data[0])],-1.0,1.0)) #2X1 mat
b1=tf.Variable(tf.zeros([2])) #1X2 mat
b2=tf.Variable(tf.zeros([1])) #1X1 mat
#equation
layer1=tf.sigmoid(tf.matmul(x,theta1)+b1) #4X2 mat
layer2=tf.sigmoid(tf.matmul(layer1,theta2)+b2) #4X1 mat
cost=-tf.reduce_mean(y*tf.log(layer2)+(1.-y)*tf.log(1.-layer2))
#black box
a = tf.Variable(0.3)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)
#init
sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)
#learning
for i in range(120000):
    sess.run(train,feed_dict={x:x_data,y:y_data})
    print(sess.run(cost,feed_dict={x:x_data,y:y_data}))
import tensorflow as tf
#input
x_data=[[1.,1.],[0.,0.],[0.,1.],[1.,0.]] #4X2 mat
y_data=[[0.],[0.],[1.],[1.]]         #4X1 mat
x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)
theta1=tf.Variable(tf.random_uniform([2,2],-1.0,1.0)) #2X2 mat
theta2=tf.Variable(tf.random_uniform([2,2],-1.0,1.0)) #2X2 mat
theta3=tf.Variable(tf.random_uniform([2,2],-1.0,1.0)) #2X2 mat
theta4=tf.Variable(tf.random_uniform([2,2],-1.0,1.0)) #2X2 mat
theta5=tf.Variable(tf.random_uniform([2,2],-1.0,1.0)) #2X2 mat
theta6=tf.Variable(tf.random_uniform([2,1],-1.0,1.0)) #2X1 mat
b1=tf.Variable(tf.zeros([2])) #1X2 mat
b2=tf.Variable(tf.zeros([2])) #1X2 mat
b3=tf.Variable(tf.zeros([2])) #1X2 mat
b4=tf.Variable(tf.zeros([2])) #1X2 mat
b5=tf.Variable(tf.zeros([2])) #1X2 mat
b6=tf.Variable(tf.zeros([1])) #1X1 mat
#equation
layer1=tf.nn.relu(tf.matmul(x,theta1)+b1) #4X2 mat
layer2=tf.nn.relu(tf.matmul(layer1,theta2)+b2) #4X2 mat
layer3=tf.nn.relu(tf.matmul(layer2,theta3)+b3) #4X2 mat
layer4=tf.nn.relu(tf.matmul(layer3,theta4)+b4) #4X2 mat
layer5=tf.nn.relu(tf.matmul(layer4,theta5)+b5) #4X2 mat
layer6=tf.sigmoid(tf.matmul(layer5,theta6)+b6) #4X1 mat
cost=-tf.reduce_mean(y*tf.log(layer6)+(1.-y)*tf.log(1.-layer6))
#black box
a = tf.Variable(0.2)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)
#init
sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)
#learning
for i in range(20000):
    sess.run(train,feed_dict={x:x_data,y:y_data})
    print(sess.run(cost,feed_dict={x:x_data,y:y_data}))
import tensorflow as tf
#input
x_data=[[1.,1.],[0.,0.],[0.,1.],[1.,0.]] #4X2 mat
y_data=[[0.],[0.],[1.],[1.]]         #4X1 mat
x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)
k=5
theta=tf.Variable(tf.random_uniform([k,2,2],-1.0,1.0)) #2X2 mat
Theta=tf.Variable(tf.random_uniform([2,1],-1.0,1.0)) #2X1 mat
b=tf.Variable(tf.zeros([k,2])) #1X2 mat
b6=tf.Variable(tf.zeros([1])) #1X1 mat
#equation
layer=[]
for i in range(k):
    layer.append(tf.nn.relu(tf.matmul(x,theta[i])+b[i])) #4X2 mat
Layer=tf.sigmoid(tf.matmul(layer[-1],theta[-1])+b6) #4X1 mat
cost=-tf.reduce_mean(y*tf.log(Layer)+(1.-y)*tf.log(1.-Layer))
#black box
a = tf.Variable(0.01)
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
import tensorflow as tf
sess=tf.Session()
x_data=[[1,2,1],[1,3,2],[1,3,4],[1,5,5],[1,7,5],[1,2,5]]
y_data=[[0],[0],[0],[1],[1],[1]]
#for i in range(10):
 #   x_data.append([1,i,i*i/10])
  #  y_data.append([i//5==1])
x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)
theta=tf.Variable(tf.random_uniform([1,len(x_data[0])],-1.0,1.0))
h=tf.matmul(theta,tf.transpose(x))
H=tf.div(1.,1+tf.exp(-h))
J=-tf.reduce_mean(tf.log(1-H)*(1-y)+tf.log(H)*y)
a=tf.Variable(0.1)
opt=tf.train.GradientDescentOptimizer(a)
train=opt.minimize(J)
init=tf.global_variables_initializer()
sess.run(init)
for i in range(15000):
    sess.run(train,feed_dict={x:x_data, y:y_data})
    print(sess.run(J,feed_dict={x:x_data, y:y_data}),sess.run(theta))
print(sess.run(H,feed_dict={x:[[1,-5.,-5.],[1,9.,9.]]}))
for i in range(10):
    print(sess.run(J,feed_dict={x:x_data,y:y_data}))
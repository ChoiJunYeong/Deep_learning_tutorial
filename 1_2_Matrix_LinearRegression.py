import tensorflow as tf
x_data=[]
y_data=[]
for i in range(10):
    y_data.append(0.01*i*i+3*i/30)
m=len(y_data)
k=5
for j in range(k):
    x_sub=[]
    for i in range(m):
        x_sub.append(pow(i,j)*0.7/pow(10,j))
    x_data.append(x_sub)
W=tf.Variable(tf.random_uniform([1,k],-1.0,1.0))
b=tf.Variable(tf.random_uniform([1],-1.0,1.0))
H=tf.matmul(W,x_data)+b
J=tf.reduce_mean(tf.square(H-y_data))
a=tf.Variable(0.5)
opt=tf.train.GradientDescentOptimizer(a)
train=opt.minimize(J)
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
for i in range(len(x_data)):
    print(x_data[i],'\n')
for i in range(900000):
    sess.run(train)
    print(sess.run(J),sess.run(W),sess.run(b))
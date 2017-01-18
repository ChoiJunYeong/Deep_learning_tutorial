import tensorflow as tf
x=[]
y=[]
for i in range(10):
    y.append([(0.3*i+15+0.01*i*i)/100])
    x_sub=[]
    for k in range(3):
        if k==0:
            x_sub.append(1)
        else:
            x_sub.append(pow(i,k)/pow(10,(k)*2))
    x.append(x_sub)
sess=tf.Session()
xt=tf.transpose(x)
xtx=tf.matmul(xt,x)
xtxxt=tf.matmul(tf.matrix_inverse(xtx),xt)
print(sess.run(xtxxt))
theta=tf.matmul(xtxxt,y)
init=tf.global_variables_initializer()
sess.run(init)
print(sess.run(theta))
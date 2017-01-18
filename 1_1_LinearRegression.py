import tensorflow as tf
x1_data=[]
x2_data=[]
y_data=[]
sess=tf.Session()
theta=[tf.Variable(0.124649),tf.Variable(0.440648),tf.Variable(0.864204)]
x1=tf.placeholder(tf.float32)
x2=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)
for i in range(1,100):
    x1_data.append(0.1*i/10)
    x2_data.append(0.01*i*i/100)
    y_data.append((0.3*i+15+0.01*i*i)/100)
H=theta[0]+theta[1]*x1+theta[2]*x2
J=tf.reduce_mean(tf.square(H-y))/2.0
init=tf.global_variables_initializer()
sess.run(init)
descent1=theta[0]-2*tf.reduce_mean(H-y)
descent2=theta[1]-2*tf.reduce_mean((H-y)*x1)
descent3=theta[2]-2*tf.reduce_mean((H-y)*x2)
update1=theta[0].assign(descent1)
update2=theta[1].assign(descent2)
update3=theta[2].assign(descent3)
#print(x1_data,'\n',x2_data,'\n',y_data,'\n')
for i in range(1,10000):
    print(sess.run(J,feed_dict={x1:x1_data,x2:x2_data,y:y_data}),sess.run(theta[0]),sess.run(theta[1]),sess.run(theta[2]))
    sess.run(update1,feed_dict={x1:x1_data,x2:x2_data,y:y_data})
    sess.run(update2,feed_dict={x1:x1_data,x2:x2_data,y:y_data})
    sess.run(update3,feed_dict={x1:x1_data,x2:x2_data,y:y_data})
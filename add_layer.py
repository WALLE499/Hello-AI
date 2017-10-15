#定义 add_layer
import tensorflow as tf
import numpy as np
#定义添加神经层的函数 add_layer()
def add_layer(inputs, in_size, out_size, activation_function = None):
    #Weights的形状,怎么确定？？？
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    #避免biases的初始值是0，所以加上0.1
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs
#[:, np.newaxis]增加维度？
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
#噪点noise,方差0.05，格式和x_data相同
noise = np.random.normal(0, 0.05, x_data.shape)
#为了接近真实情况，所以加上噪点noise
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

#定义输入神经元有一个，隐藏层有10个，输出神经元有一个的神经网络
layer1 = add_layer(xs, 1, 10, activation_function = tf.nn.relu)
prediction = add_layer(layer1, 10, 1, activation_function = None)
#损失函数
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                    reduction_indices = [1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#变量初始化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
#开始训练
for i in range(1000):
    sess.run(train_step, feed_dict = {xs:x_data, ys:y_data})
    if i % 50 == 0:
        print(sess.run(loss,feed_dict={xs:x_data, ys:y_data}))

        


    

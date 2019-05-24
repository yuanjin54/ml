import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# 参数
learning_rate = 0.1
input_size = 784
output_size = 10
batch_size = 50

# 导入数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(x, w_h, w_o):
    h = tf.nn.sigmoid(tf.matmul(x, w_h))
    return tf.matmul(h, w_o)


# 输入训练数据
x_data = tf.placeholder(tf.float32, [None, input_size])
y_data = tf.placeholder(tf.float32, [None, output_size])

w_h = init_weights([input_size, 625])
w_o = init_weights([625, output_size])

y_out = model(x_data, w_h, w_o)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_out, labels=y_data))

train_model = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

y_pre = tf.argmax(y_out, 1)

g_x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
g_y = np.array(range(11))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX) + 1, 128)):
            sess.run(train_model, feed_dict={x_data: trX[start:end], y_data: trY[start:end]})
        if (i + 1) % 10 == 0:
            loss_out = np.mean(np.argmax(teY, axis=1) == sess.run(y_pre, feed_dict={x_data: teX}))
            print(i + 1, loss_out)
            g_y[(i + 1) % 10] = loss_out
    print(g_x, g_y)

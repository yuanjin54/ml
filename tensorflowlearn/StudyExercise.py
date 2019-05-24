import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
batch_size = 50

# 添加层
def add_layer(inputs, in_size, out_size, activation_fun=None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    bias = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    out = tf.matmul(inputs, weights) + bias
    if activation_fun is None:
        return out
    else:
        return activation_fun(out)


# 数据传入
x_data = tf.placeholder(tf.float32, [None, 784])
y_data = tf.placeholder(tf.float32, [None, 10])

# 构建模型(一层神经网络)
output = add_layer(x_data, 784, 10, activation_fun=tf.nn.softmax)
loss = tf.reduce_mean(-tf.reduce_sum(y_data * tf.log(output), reduction_indices=[1]))
# 训练优化器让loss达到最小值
model = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# 初始化全局参数
sess = tf.Session()
sess.run(tf.initialize_all_variables())


def compute_accuracy(images, labels):
    global output
    y_pre = sess.run(output, feed_dict={x_data: images})
    correct = tf.equal(tf.argmax(y_pre, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return sess.run(accuracy, feed_dict={x_data: images, y_data: labels})


for i in range(10000):
    x_batch, y_batch = mnist.train.next_batch(100)
    sess.run(model, feed_dict={x_data: x_batch, y_data: y_batch})
    if i % 100 == 0:
        res = compute_accuracy(mnist.test.images, mnist.test.labels)
        print(res)
sess.close()

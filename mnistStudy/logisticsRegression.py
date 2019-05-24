import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 参数
learning_rate = 0.01
input_size = 784
output_size = 10
batch_size = 50

# 导入数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 输入训练数据
x_data = tf.placeholder(tf.float32, [None, input_size])
y_data = tf.placeholder(tf.float32, [None, output_size])

# 构建 logistics regression 模型 (-----start-----) #
w = tf.Variable(tf.zeros([input_size, output_size]))
b = tf.Variable(tf.zeros([1, output_size]))
# 选择 softmax 构建逻辑回归模型
y_pre = tf.nn.softmax(tf.matmul(x_data, w) + b)
# 交叉熵损失函数
loss = tf.reduce_mean(-tf.reduce_sum(y_data * tf.log(y_pre), reduction_indices=[1]))
# 选择梯度下降模型训练loss
model = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
# 构建 logistics regression 模型 (-----end-----) #

# 开始训练
with tf.Session() as sess:
    # 初始化所有变量
    sess.run(tf.initialize_all_variables())
    for epoch in range(25):
        avg_loss = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(model, feed_dict={x_data: x_batch, y_data: y_batch})
            avg_loss += sess.run(loss, feed_dict={x_data: x_batch, y_data: y_batch}) / total_batch
            if (epoch + 1) % 10 == 0:
                print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(avg_loss))
    print("finished")
    # 测试计算正确率
    correct = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y_data, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    print("正确率：", accuracy.eval({x_data: mnist.test.images, y_data: mnist.test.labels}))

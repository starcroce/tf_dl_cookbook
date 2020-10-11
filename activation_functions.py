import matplotlib.pyplot as plt
import numpy
import tensorflow as tf


def threshold(x):
    cond = tf.less(x, tf.zeros(tf.shape(x), dtype=x.dtype))
    out = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))
    return out


h = numpy.linspace(-1, 1, 50)
out = threshold(h)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    y = sess.run(out)

    plt.xlabel("Activity of Neuron")
    plt.ylabel("Output of Neuron")
    plt.title("Threshold Activation Function")
    plt.plot(h, y)
    plt.show()


h = numpy.linspace(-10, 10, 50)
out = tf.sigmoid(h)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    y = sess.run(out)

    plt.xlabel("Activity of Neuron")
    plt.ylabel("Output of Neuron")
    plt.title("Sigmoid Activation Function")
    plt.plot(h, y)
    plt.show()


h = numpy.linspace(-10, 10, 50)
out = tf.tanh(h)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    y = sess.run(out)

    plt.xlabel("Activity of Neuron")
    plt.ylabel("Output of Neuron")
    plt.title("Sigmoid Activation Function")
    plt.plot(h, y)
    plt.show()


b = tf.Variable(tf.random_normal([1, 1], stddev=2))
w = tf.Variable(tf.random_normal([3, 1], stddev=2))
X_in = tf.Variable(tf.random_normal([1, 3], stddev=2))
linear_out = tf.matmul(X_in, w) + b
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    out = sess.run(linear_out)
    print(out)


h = numpy.linspace(-10, 10, 50)
out = tf.nn.relu(h)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    y = sess.run(out)

    plt.xlabel("Activity of Neuron")
    plt.ylabel("Output of Neuron")
    plt.title("Relu Activation Function")
    plt.plot(h, y)
    plt.show()


h = numpy.linspace(-5, 5, 50)
out = tf.nn.softmax(h)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    y = sess.run(out)

    plt.xlabel("Activity of Neuron")
    plt.ylabel("Output of Neuron")
    plt.title("Softmax Activation Function")
    plt.plot(h, y)
    plt.show()

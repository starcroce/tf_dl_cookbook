import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def normalize(X):
    mean = np.mean(X)
    std = np.std(X)
    X = (X - mean) / std
    return X


def append_bias_reshape(features, labels):
    m = features.shape[0]
    n = features.shape[1]
    x = np.reshape(np.c_[np.ones(m), features], [m, n + 1])
    y = np.reshape(labels, [m, 1])
    return x, y


boston = tf.contrib.learn.datasets.load_dataset("boston")
X_train, Y_train = boston.data, boston.target
X_train = normalize(X_train)
X_train, Y_train = append_bias_reshape(X_train, Y_train)
m = len(X_train)  # number of training examples
n = 13 + 1  # number of features + bias

X = tf.placeholder(tf.float32, name="X", shape=[m, n])
Y = tf.placeholder(tf.float32, name="Y")

w = tf.Variable(tf.random_normal([n, 1]))
b = tf.Variable(0.0)
Y_hat = tf.matmul(X, w) + b

loss = tf.reduce_mean(tf.square(Y - Y_hat), name="loss")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

init_op = tf.global_variables_initializer()
total = []

with tf.Session() as sess:
    sess.run([init_op])
    writer = tf.summary.FileWriter("graph", sess.graph)

    for i in range(100):
        _, l = sess.run([optimizer, loss], feed_dict={X: X_train, Y: Y_train})
        total.append(l)
        print(f"Epoch {i}: Loss {l}")

    writer.close()
    w_value, b_value = sess.run([w, b])

plt.plot(total)
plt.show()

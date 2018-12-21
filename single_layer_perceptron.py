import tensorflow as tf
import numpy as np

eta = 0.4  # learning rate
epsilon = 1e-3
max_epochs = 100


def threshold(x):
    cond = tf.less(x, tf.zeros(tf.shape(x), dtype=x.dtype))
    out = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))
    return out


# Training Data Y = AB + BC, sum of two linear functions.
T, F = 1.0, 0.0
X_in = [
    [T, T, T, T],
    [T, T, F, T],
    [T, F, T, T],
    [T, F, F, T],
    [F, T, T, T],
    [F, T, F, T],
    [F, F, T, T],
    [F, F, F, T],
]
Y = [
    [T], [T], [F], [F], [T], [F], [F], [F],
]

W = tf.Variable(tf.random_normal([4, 1], stddev=2, seed=0))
h = tf.matmul(X_in, W)
Y_hat = threshold(h)

error = Y - Y_hat
mean_error = tf.reduce_mean(tf.square(error))
dW = eta * tf.matmul(X_in, error, transpose_a=True)
train = tf.assign(W, W + dW)

init = tf.global_variables_initializer()
err = 1
epoch = 0

with tf.Session() as sess:
    sess.run(init)
    while err > epsilon and epoch < max_epochs:
        epoch += 1
        err, _ = sess.run([mean_error, train])
        print(f'epoch {epoch} mean error: {err}')

    print('Training complete')

import tensorflow as tf

m = 1000
n = 15
P = 2

X = tf.placeholder(tf.float32, name='X', shape=[m, n])
Y = tf.placeholder(tf.float32, name='Y', shape=[m, P])

w0 = tf.Variable(tf.zeros([1, P]), name='bias')
w1 = tf.Variable(tf.random_normal([n, 1]), name='weights')

Y_hat = tf.matmul(X, w1) + w0

entropy = tf.nn.softmax_cross_entropy_with_logits(Y_hat, Y)
loss = tf.reduce_mean(entropy)

lmbd = tf.constant(0.8)
l1_reg = lmbd * tf.reduce_sum(tf.abs(w1))
l2_reg = lmbd * tf.nn.l2_loss(w1)

loss += l2_reg

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data/mnist/', one_hot=True)


def train_size(num):
    print(f'Total training images in dataset is {mnist.train.images.shape}')
    x_train = mnist.train.images[:num, :]
    y_train = mnist.train.labels[:num, :]

    print(f'x_train examples loaded {x_train.shape}')
    print(f'y_train examples loaded {y_train.shape}')
    return x_train, y_train


def test_size(num):
    print(f'Total test images in dataset is {mnist.test.images.shape}')
    x_test = mnist.test.images[:num, :]
    y_test = mnist.test.labels[:num, :]

    print(f'x_test examples loaded {x_test.shape}')
    print(f'y_test examples loaded {y_test.shape}')
    return x_test, y_test


def display_digit(num, x_train, y_train):
    print(y_train[num])
    label = y_train[num].argmax(axis=0)
    image = x_train[num].reshape([28, 28])

    plt.title(f'Example {num}, label {label}')
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()


def display_multi_flat(start, stop, x_train):
    images = x_train[start].reshape([1, 784])
    for i in range(start + 1, stop):
        images = np.concatenate((images, x_train[i].reshape([1, 784])))

    plt.imshow(images, cmap=plt.get_cmap('gray_r'))
    plt.show()


# get training data and show some random examples
x_train, y_train = train_size(50000)
display_digit(np.random.randint(0, x_train.shape[0]), x_train, y_train)
display_multi_flat(0, 400, x_train)

# hyper parameters and network parameters
learning_rate = 0.001
training_iters = 500
batch_size = 128
dropout = 0.85
display_step = 10
n_inputs = 784
n_classes = 10

x = tf.placeholder(tf.float32, [None, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)


def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def conv_net(x, weights, biases, dropout):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)

    # reshape cov2 output to match the input of fully connected layer
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


weights = {
    # 5 * 5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5 * 5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected layer, 7 * 7 * 64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
    # output layer, 1024 inputs, 10 outputs for digits
    'out': tf.Variable(tf.random_normal([1024, n_classes])),
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes])),
}

pred = conv_net(x, weights, biases, keep_prob)
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
corrected_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(corrected_prediction, tf.float32))
init = tf.global_variables_initializer()

train_loss = []
train_acc = []
test_acc = []
with tf.Session() as sess:
    sess.run(init)
    step = 1

    while step <= training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(
            optimizer,
            feed_dict={x: batch_x, y: batch_y, keep_prob: dropout}
        )
        if step % display_step == 0:
            loss_train, acc_train = sess.run(
                [cost, accuracy],
                feed_dict={x: batch_x, y: batch_y, keep_prob: 1}
            )
            print(
                f'Iter {step}, '
                f'minibatch loss: {loss_train}, '
                f'training accuracy {acc_train}'
            )

            acc_test = sess.run(
                accuracy,
                feed_dict={x: mnist.test.images,
                           y: mnist.test.labels, keep_prob: 1}
            )
            print(f'Test accuracy: {acc_test}')

            train_loss.append(loss_train)
            train_acc.append(acc_train)
            test_acc.append(acc_test)

        step += 1


eval_indices = range(0, training_iters, display_step)
plt.plot(eval_indices, train_loss, 'k-')
plt.title('Softmax loss per iteration')
plt.xlabel('Iteration')
plt.ylabel('Softmax loss')
plt.show()

plt.plot(eval_indices, train_acc, 'k-', label='Train set accuracy')
plt.plot(eval_indices, test_acc, 'r--', label='Train set accuracy')
plt.title('Train and test accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

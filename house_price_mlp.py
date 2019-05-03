import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.layers as layers
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

boston = datasets.load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['target'] = boston.target

X_train, X_test, y_train, y_test = train_test_split(
    df[['RM', 'LSTAT', 'PTRATIO']], df[['target']], test_size=0.3, random_state=0)
X_train = MinMaxScaler().fit_transform(X_train)
X_test = MinMaxScaler().fit_transform(X_test)
y_train = MinMaxScaler().fit_transform(y_train)
y_test = MinMaxScaler().fit_transform(y_test)

m = len(X_train)
n = 3  # number of features
n_hidden = 20  # number of hidden neurons
batch_size = 200
eta = 0.01
max_epoch = 1000


def multilayer_perceptron(x):
    fcl = layers.fully_connected(
        x, n_hidden, activation_fn=tf.nn.relu, scope='fcl')
    out = layers.fully_connected(
        fcl, 1, activation_fn=tf.nn.sigmoid, scope='out')
    return out


# build model, loss and train op
x = tf.placeholder(tf.float32, name='X', shape=[m, n])
y = tf.placeholder(tf.float32, name='Y')
y_hat = multilayer_perceptron(x)
correct_prediction = tf.square(y - y_hat)
mse = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
train = tf.train.AdamOptimizer(learning_rate=eta).minimize(mse)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('graphs', sess.graph)
    for i in range(max_epoch):
        _, l, p = sess.run([train, mse, y_hat], feed_dict={
            x: X_train, y: y_train})
        if i % 100 == 0:
            print(f'Epoch {i}: Loss {l}')

    print('Training Done!')
    correct_prediction = tf.square(y - y_hat)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    print('Mean square error:', accuracy.eval({x: X_train, y: y_train}))
    plt.scatter(y_train, p)
    plt.show()
    writer.close()

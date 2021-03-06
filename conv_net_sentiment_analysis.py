import tensorflow as tf
import tflearn

from tflearn.data_utils import pad_sequences, to_categorical
from tflearn.datasets import imdb
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.core import dropout, fully_connected, input_data
from tflearn.layers.estimator import regression
from tflearn.layers.merge_ops import merge

train, test, _ = imdb.load_data(
    path="./data/imdb.pkl", n_words=10000, valid_portion=0.1
)
train_x, train_y = train
test_x, test_y = test
train_x = pad_sequences(train_x, maxlen=100, value=0)
test_x = pad_sequences(test_x, maxlen=100, value=0)
train_y = to_categorical(train_y, nb_classes=2)
test_y = to_categorical(test_y, nb_classes=2)

print("train_x size", train_x.size)
print("train_y size", train_y.size)
print("test_x size", test_x.size)
print("test_y size", test_y.size)

network = input_data(shape=[None, 100], name="input")
network = tflearn.embedding(network, input_dim=10000, output_dim=128)

branch_1 = conv_1d(
    network, 128, 3, padding="valid", activation="relu", regularizer="L2"
)
branch_2 = conv_1d(
    network, 128, 4, padding="valid", activation="relu", regularizer="L2"
)
branch_3 = conv_1d(
    network, 128, 5, padding="valid", activation="relu", regularizer="L2"
)

network = merge([branch_1, branch_2, branch_3], mode="concat", axis=1)
network = tf.expand_dims(network, 2)
network = global_max_pool(network)
network = dropout(network, 0.5)
network = fully_connected(network, 2, activation="softmax")

network = regression(
    network,
    optimizer="adam",
    learning_rate=0.001,
    loss="categorical_crossentropy",
    name="target",
)

model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(
    train_x,
    train_y,
    n_epoch=5,
    shuffle=True,
    validation_set=(test_x, test_y),
    show_metric=True,
    batch_size=32,
)

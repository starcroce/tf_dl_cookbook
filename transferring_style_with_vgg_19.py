import os
import sys

import matplotlib as plt
import numpy as np
import scipy.io as sio
import scipy.misc
import tensorflow as tf

from matplotlib.pyplot import imshow
from PIL import Image
from scipy import io as sio

OUTPUT_DIR = "output/"
STYLE_IMAGE = "data/starry_night.jpg"
CONTENT_IMAGE = "data/marilyn_monroe_in_1952.jpg"

NOISE_RATIO = 0.6  # how much noise is in the image
BETA = 5  # how much emphasis in the content loss
ALPHA = 100  # how much emphasis in the style loss

# http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat
VGG_MODEL = "data/imagenet-vgg-verydeep-19.mat"
MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))

content_image = scipy.misc.imread(CONTENT_IMAGE)
style_image = scipy.misc.imread(STYLE_IMAGE)
target_shape = content_image.shape
style_image = scipy.misc.imresize(style_image, target_shape)
scipy.misc.imsave(STYLE_IMAGE, style_image)


def load_vgg_model(path, image_height, image_width, color_channels):
    vgg = scipy.io.loadmat(path)
    vgg_layers = vgg["layers"]

    def _weights(layer, expected_layer_name):
        """Return the weights and bias from the VGG model for a given layer."""

        W = vgg_layers[0][layer][0][0][2][0][0]
        b = vgg_layers[0][layer][0][0][2][0][1]
        layer_name = vgg_layers[0][layer][0][0][0][0]
        assert (
            layer_name == expected_layer_name
        ), f"Expected {expected_layer_name}, got {layer_name}"
        return W, b

    def _relu(conv2d_layer):
        """Return the relu function wrapped over a tensorflow layer."""

        return tf.nn.relu(conv2d_layer)

    def _conv2d(prev_layer, layer, layer_name):
        """Return the conv2d layer using the weights, bias from the VGG model `layer`."""

        W, b = _weights(layer, layer_name)
        W = tf.constant(W)
        b = tf.constant(np.reshape(b, (b.size)))
        return (
            tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding="SAME") + b
        )

    def _conv2d_relu(prev_layer, layer, layer_name):
        """Return the conv2d + relu layer using the weights, bias from the VGG model `layer`."""

        return _relu(_conv2d(prev_layer, layer, layer_name))

    def _avgpool(prev_layer):
        """return average pooling layer."""

        return tf.nn.avg_pool(
            prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
        )

    # construct the graph model
    graph = {}
    graph["input"] = tf.Variable(
        np.zeros((1, image_height, image_width, color_channels)), dtype="float32"
    )
    graph["conv1_1"] = _conv2d_relu(graph["input"], 0, "conv1_1")
    graph["conv1_2"] = _conv2d_relu(graph["conv1_1"], 2, "conv1_2")
    graph["avgpool1"] = _avgpool(graph["conv1_2"])
    graph["conv2_1"] = _conv2d_relu(graph["avgpool1"], 5, "conv2_1")
    graph["conv2_2"] = _conv2d_relu(graph["conv2_1"], 7, "conv2_2")
    graph["avgpool2"] = _avgpool(graph["conv2_2"])
    graph["conv3_1"] = _conv2d_relu(graph["avgpool2"], 10, "conv3_1")
    graph["conv3_2"] = _conv2d_relu(graph["conv3_1"], 12, "conv3_2")
    graph["conv3_3"] = _conv2d_relu(graph["conv3_2"], 14, "conv3_3")
    graph["conv3_4"] = _conv2d_relu(graph["conv3_3"], 16, "conv3_4")
    graph["avgpool3"] = _avgpool(graph["conv3_4"])
    graph["conv4_1"] = _conv2d_relu(graph["avgpool3"], 19, "conv4_1")
    graph["conv4_2"] = _conv2d_relu(graph["conv4_1"], 21, "conv4_2")
    graph["conv4_3"] = _conv2d_relu(graph["conv4_2"], 23, "conv4_3")
    graph["conv4_4"] = _conv2d_relu(graph["conv4_3"], 25, "conv4_4")
    graph["avgpool4"] = _avgpool(graph["conv4_4"])
    graph["conv5_1"] = _conv2d_relu(graph["avgpool4"], 28, "conv5_1")
    graph["conv5_2"] = _conv2d_relu(graph["conv5_1"], 30, "conv5_2")
    graph["conv5_3"] = _conv2d_relu(graph["conv5_2"], 32, "conv5_3")
    graph["conv5_4"] = _conv2d_relu(graph["conv5_3"], 34, "conv5_4")
    graph["avgpool5"] = _avgpool(graph["conv5_4"])

    return graph


def content_loss_func(sess, model):
    """Content loss function defined in the paper."""

    def _content_loss(p, x):
        # N is the number of filters at layer 1
        N = p.shape[3]
        # M is the height * width of the feature map at layer 1
        M = p.shape[1] * p.shape[2]
        return (1 / (4 * N * M)) * tf.reduce_mean(tf.pow(x - p, 2))

    return _content_loss(sess.run(model["conv4_2"]), model["conv4_2"])


# define the VGG layers that we are going to reuse, if we would like to have softer features
# we will need to increase the weight of higher layers (conv5_1)
# and decrease the weight of lower layers (conv1_1)
STYLE_LAYERS = [
    ("conv1_1", 0.5),
    ("conv2_1", 1.0),
    ("conv3_1", 1.5),
    ("conv4_1", 3.0),
    ("conv5_1", 4.0),
]


def style_loss_func(sess, model):
    """Style loss function defined in the paper."""

    def _gram_matrix(F, N, M):
        ft = tf.reshape(F, (M, N))
        return tf.matmul(tf.transpose(ft), ft)

    def _style_loss(a, x):
        # N is the number of filters at layer 1
        N = a.shape[3]
        # M is the height * width of the feature map at layer 1
        M = a.shape[1] * a.shape[2]
        # A is the style representation of the original image at layer 1
        A = _gram_matrix(a, N, M)
        # G is the style representation of generated image at layer 1
        G = _gram_matrix(x, N, M)
        return (1 / (4 * N ** 2 * M ** 2)) * tf.reduce_sum(tf.pow(G - A, 2))

    E = [
        _style_loss(sess.run(model[layer_name]), model[layer_name])
        for layer_name, _ in STYLE_LAYERS
    ]
    W = [w for _, w in STYLE_LAYERS]
    return sum([W[l] * E[l] for l in range(len(STYLE_LAYERS))])


def generate_noise_image(content_image, noise_ratio=NOISE_RATIO):
    """Returns a noise image intermixed with a content image at a certain ratio."""

    noise_image = np.random.uniform(
        -20,
        20,
        (
            1,
            content_image[0].shape[0],
            content_image[0].shape[1],
            content_image[0].shape[2],
        ),
    ).astype("float32")

    # white noise image from the content representation and take a weighted average
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)
    return input_image


def process_image(image):
    # resize the image for convent input, there is no change but just adding a new dim
    image = np.reshape(image, ((1,) + image.shape))
    # input to the VGG model, expects the mean to be subtracted
    image = image - MEAN_VALUES
    return image


def load_image(path):
    image = scipy.misc.imread(path)
    return process_image(image)


def save_image(path, image):
    # output should add back the mean
    image = image + MEAN_VALUES
    # remove the first useless dimension
    image = image[0]
    image = np.clip(image, 0, 255).astype("uint8")
    scipy.misc.imsave(path, image)


sess = tf.InteractiveSession()
content_image = load_image(CONTENT_IMAGE)
style_image = load_image(STYLE_IMAGE)

model = load_vgg_model(
    VGG_MODEL,
    style_image[0].shape[0],
    style_image[0].shape[1],
    style_image[0].shape[2],
)
input_image = generate_noise_image(content_image)
sess.run(tf.initialize_all_variables())

# construct content loss using content image
sess.run(model["input"].assign(content_image))
content_loss = content_loss_func(sess, model)

# construct style loss using style image
sess.run(model["input"].assign(style_image))
style_loss = style_loss_func(sess, model)

# construct total loss as a weighted combination of content loss and style loss
total_loss = BETA * content_loss + ALPHA * style_loss

optimizer = tf.train.AdamOptimizer(2.0)
train_step = optimizer.minimize(total_loss)

sess.run(tf.initialize_all_variables())
sess.run(model["input"].assign(input_image))

ITERATIONS = 1000
for it in range(ITERATIONS):
    sess.run(train_step)
    if it % 100 == 0:
        mixed_image = sess.run(model["input"])
        print(
            f"Iteration: {it}, "
            f"Sum: {sess.run(tf.reduce_sum(mixed_image))}, "
            f"Cost: {sess.run(total_loss)}"
        )
        if not os.path.exists(OUTPUT_DIR):
            os.mkdir(OUTPUT_DIR)
        filename = f"{OUTPUT_DIR}{it}.png"
        save_image(filename, mixed_image)

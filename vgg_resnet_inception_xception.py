import matplotlib.pyplot as plt
import numpy as np
from keras.applications import (VGG16, VGG19, InceptionV3, ResNet50, Xception,
                                imagenet_utils)
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img
from matplotlib.pyplot import imshow
from PIL import Image

MODELS = {
    'vgg16': (VGG16, (224, 224)),
    'vgg19': (VGG19, (224, 224)),
    'inception': (InceptionV3, (299, 299)),
    'xception': (Xception, (299, 299)),
    'resnet': (ResNet50, (224, 224)),
}

WEIGHTS = {
    'vgg16': './data/vgg16_weights_tf_dim_ordering_tf_kernels.h5',
    'vgg19': './data/vgg19_weights_tf_dim_ordering_tf_kernels.h5',
    'inception': './data/inception_v3_weights_tf_dim_ordering_tf_kernels.h5',
    'xception': './data/xception_weights_tf_dim_ordering_tf_kernels.h5',
    'resnet': './data/resnet50_weights_tf_dim_ordering_tf_kernels.h5',
}


def image_load_and_convert(image_path, model):
    pil_im = Image.open(image_path, 'r')
    imshow(np.asarray(pil_im))
    input_shape = MODELS[model][1]
    preprocess = imagenet_utils.preprocess_input
    image = load_img(image_path, target_size=input_shape)
    image = img_to_array(image)
    # the original models are trained on additional dimension about batch size
    # so we need to add it too to match the model input
    image = np.expand_dims(image, axis=0)
    image = preprocess(image)
    return image


def classify_image(image_path, model):
    print(image_path, model)
    img = image_load_and_convert(image_path, model)
    network = MODELS[model][0]
    model = network(weights=WEIGHTS[model])
    preds = model.predict(img)
    p = imagenet_utils.decode_predictions(preds)
    for i, (imagenet_id, label, prob) in enumerate(p[0]):
        print(f'{i + 1}. {label}: {prob * 100}%')


classify_image('./data/parrot.jpg', 'vgg16')
classify_image('./data/parrot.jpg', 'vgg19')
classify_image('./data/parrot.jpg', 'resnet')
classify_image('./data/parrot_cropped.png', 'resnet')
classify_image('./data/incredible_hulk_180.jpg', 'resnet')
classify_image('./data/panda_cropped.png', 'resnet')
classify_image('./data/space_shuttle_1.jpg', 'resnet')
classify_image('./data/space_shuttle_2.jpg', 'resnet')
classify_image('./data/space_shuttle_3.jpg', 'resnet')
classify_image('./data/space_shuttle_4.jpg', 'resnet')
classify_image('./data/parrot.jpg', 'inception')
classify_image('./data/parrot.jpg', 'xception')


def print_model(model):
    print(f'Model: {model}')
    network = MODELS[model][0]
    model = network(weights=WEIGHTS[model])
    print(model.summary())


print_model('vgg19')

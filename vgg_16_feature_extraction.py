import numpy as np

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.preprocessing import image

base_model = VGG16(weights="./data/vgg16_weights_tf_dim_ordering_tf_kernels.h5")
for i, layer in enumerate(base_model.layers):
    print(i, layer.name, layer.output_shape)
model = Model(input=base_model.input, output=base_model.get_layer("block4_pool").output)

img_path = "./data/cat.jpeg"
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
features = model.predict(x)
print(features)

from keras import backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing import image

base_model = InceptionV3(
    include_top=False, weights='./data/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
for layer in base_model.layers:
    print(layer.name, layer.input_shape, layer.output_shape)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
prediction = Dense(200, activation='softmax')(x)
model = Model(input=base_model.input, output=prediction)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
# model.fit_generator(...)

# we train the top 2 inception blocks, so freeze the first 172 layers and unfreeze the res
for layer in base_model.layers[:172]:
    layer.trainable = False
for layer in base_model.layers[172:]:
    layer.trainable = True

model.compile(
    optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
# model.fit_generator(...)

from keras import applications, optimizers
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator

img_width, img_height = 256, 256
batch_size = 16
epochs = 50

train_data_dir = "data/dogs_and_cats/train"
validation_data_dir = "data/dogs_and_cats/validation"

out_categories = 1
num_train_samples = 10000
num_validation_samples = 2500

base_model = applications.VGG16(
    weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3)
)
for layer in base_model.layers[:15]:
    layer.trainable = False

top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation="relu"))
top_model.add(Dropout(0.5))
top_model.add(Dense(out_categories, activation="sigmoid"))

model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
model.compile(
    loss="binary_crossentropy",
    optimizer=optimizers.SGD(lr=0.001, momentum=0.9),
    metrics=["accuracy"],
)

# init the train and test generators with data augmentation
train_datagen = ImageDataGenerator(rescale=1 / 255, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1 / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="binary",
)
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="binary",
    shuffle=False,
)

model.fit_generator(
    train_generator,
    steps_per_epoch=num_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=num_validation_samples // batch_size,
    verbose=2,
    workers=12,
)

score = model.evaluate_generator(
    validation_generator, num_validation_samples // batch_size
)
scores = model.predict_generator(
    validation_generator, num_validation_samples // batch_size
)

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import spacy
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers.core import Dense, Dropout, Reshape
from keras.layers.merge import concatenate
from keras.layers.recurrent import LSTM
from keras.models import Input, Model
from keras.preprocessing import image
from sklearn.externals import joblib

# demo code and data came from https://github.com/iamaaditya/VQA_Demo
label_encoder_file = './data/FULL_labelencoder_trainval.pkl'
VQA_weights_file = './data/VQA_MODEL_WEIGHTS.hdf5'
length_max_questions = 30
length_vgg_features = 4096
length_spacy_features = 300


def get_image_features(img_path, vgg16_model_full):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = vgg16_model_full.predict(x)
    model_extract_features = Model(
        input=vgg16_model_full.input, output=vgg16_model_full.get_layer('fc2').output)
    fc2_features = model_extract_features.predict(x)
    fc2_features = fc2_features.reshape((1, length_vgg_features))
    return fc2_features


def get_question_features(question):
    """Given a question, returns the time series vector with each word token
    transformed into a 300 dimension representation calculated using Glove."""

    word_embeddings = spacy.load('en_core_web_md')
    tokens = word_embeddings(question)
    n_tokens = len(tokens)
    if len(tokens) > length_max_questions:
        n_tokens = length_max_questions
    question_tensor = np.zeros((1, length_max_questions, 300))
    for j in range(n_tokens):
        question_tensor[0, j, :] = tokens[j].vector
    return question_tensor


img_file_name = './data/girl.jpg'
img_0 = PIL.Image.open(img_file_name)
model = VGG16(
    weights='./data/vgg16_weights_tf_dim_ordering_tf_kernels.h5', include_top=True)
image_features = get_image_features(img_file_name, model)
print(image_features.shape)

question = 'Who is in this picture?'
language_features = get_question_features(question)
print(language_features.shape)


def build_combine_model(
    num_of_lstm=3,
    num_of_lstm_hidden_units=512,
    num_of_dense_layer=3,
    num_of_hidden_units=1024,
    activation_func='tanh',
    dropout=0.5
):
    # input image
    input_image = Input(shape=(length_vgg_features,), name='input_image')
    model_image = Reshape(
        (length_vgg_features,),
        input_shape=(length_vgg_features,),
    )(input_image)

    # input language
    input_language = Input(
        shape=(length_max_questions, length_spacy_features), name='input_language')
    model_language = LSTM(
        num_of_lstm_hidden_units,
        return_sequences=True,
        name='lstm_1',
    )(input_language)
    model_language = LSTM(
        num_of_lstm_hidden_units,
        return_sequences=True,
        name='lstm_2',
    )(model_language)
    model_language = LSTM(
        num_of_lstm_hidden_units,
        return_sequences=False,
        name='lstm_3',
    )(model_language)

    # concatenate, dense, dropout
    model = concatenate([model_image, model_language])
    for _ in range(num_of_dense_layer):
        model = Dense(num_of_hidden_units, kernel_initializer='uniform')(model)
        model = Dropout(dropout)(model)

    model = Dense(1000, activation='softmax')(model)
    model = Model(inputs=[input_image, input_language], outputs=model)
    return model


combined_model = build_combine_model()
print(combined_model.summary())
combined_model.load_weights(VQA_weights_file)
combined_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

y_output = combined_model.predict([image_features, language_features])
for label in reversed(np.argsort(y_output)[0, -5:]):
    print(str(round(y_output[0, label] * 100, 2)).zfill(5))

# print label name from label encoder will fail because of sklearn version issue
label_encoder = joblib.load(label_encoder_file)

import time
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

model = keras.models.load_model('model_catsVSdogs_10epochs.h5')

audio_file = open('Cat-meow.mp3', 'rb')
audio_bytes = audio_file.read()

Image_Width = 128
Image_Height = 128
Image_Size = (Image_Width, Image_Height)


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))

    return s


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    Y_prediction = sigmoid((np.dot(w.T, X) + b))

    return Y_prediction


st.title("Cat Image Classification")

st.write('\n')

image = Image.open('images/image.png')
show = st.image(image, use_column_width=True)

st.sidebar.title("Upload Image")

st.set_option('deprecation.showfileUploaderEncoding', False)

uploaded_file = st.sidebar.file_uploader(" ", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    show.image(image, 'Uploaded Image', use_column_width=True)

    image = image.resize(Image_Size)
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    image = image / 255

st.sidebar.write('\n')

if st.sidebar.button("Click Here to Classify"):
    if uploaded_file is None:
        st.sidebar.write("Please upload an Image to Classify")
    else:
        with st.spinner('Classifying ...'):
            prediction = model.predict([image])[0]
            time.sleep(2)
            st.success('Done!')
            prediction = prediction[0]

        st.sidebar.header("Algorithm Predicts: ")
        cat_probability = "{:.3f}".format(float(prediction * 100))
        dog_probability = "{:.3f}".format(float(100 - prediction * 100))

        if prediction > 0.5:
            st.sidebar.write("It's a 'Cat' picture.", '\n')
            st.sidebar.write('**Probability: **', cat_probability, '%')
            st.sidebar.audio(audio_bytes)
        else:
            st.sidebar.write(" It's a 'Dog' picture ", '\n')
            st.sidebar.write('**Probability: **', dog_probability, '%')

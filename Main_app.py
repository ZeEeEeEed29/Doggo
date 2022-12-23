
import numpy as np
import streamlit as st 
import cv2 
from keras.models import load_model

model = load_model('dog_breed.h5')

CLASS_NAMES = ['scottish_deerhound','maltese_dog','afghan_hound','entlebucher','bernese_mountain_dog']


st.title('Dog Breed Prediction')

st.markdown('Upload an image of the dog')

dog_image = st.file_uploader ('Choose an image...', type = 'png')




submit = st.button('Predict')

if submit :

    if dog_image is not None:

        file_bytes = np.asarray(bytearray(dog_image.read()), dtype = np.uint8)

        opencv_image = cv2.imdecode(file_bytes, 1)

        st.image(opencv_image,channels = 'BGR')
    
        opencv_image = cv2.resize(opencv_image,(224,224))

        opencv_image.shape = (1,224,224,3)

        Y_pred = model.predict(opencv_image)

        st.title(str('The dog breed is'+ CLASS_NAMES[np.argmax(Y_pred)]))
    else: 
        st.write("Ahoy! Put a dog image in there mate")

st.header("How did this model do?")
rate = st.slider("Rate it!",1,10)
if st.button("Slider"):
    st.write(rate)
import streamlit as st
import numpy as np
import tensorflow as tf
import keras 
import io
import os
import sys
import cv2
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array


#def dodgeV2(x, y):
    #return cv2.divide(x, 64 - y, scale=64)

#def pencilsketch(inp_img):
    #img_gray = cv2.cvtColor(inp_img, cv2.COLOR_BGR2GRAY)
    #img_invert = cv2.bitwise_not(img_gray)
    #img_smoothing = cv2.GaussianBlur(img_invert, (21, 21),sigmaX=0, sigmaY=0)
    #final_img = dodgeV2(img_gray, img_smoothing)
    #return(final_img)


st.title("Clasificador de Imagenes App")
st.write("Esta App es para clasificar imagenes en sus respectivas clases")

file_image = st.sidebar.file_uploader("Sube tu imagen a clasificar", type=['jpeg','jpg','png'])
image=file_image

longitud, altura = 64,64
modelo = r"d:/udenar 8vo/PRO AUTO Y CONTROL/laPoderosa (1).h5"
model = load_model(modelo)

if file_image is None:
    st.write("No ha subido ninguna imagen")

else:
    input_img = Image.open(file_image)
    target_size=(longitud, altura)
    #final_sketch = pencilsketch(np.array(input_img))
    st.write("**Imagen de entrada**")
    st.image(input_img, use_column_width=True)
    def predict(file):
        image = Image.open(file)
        input_arr = keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # Convert single image to a batch.
        predictions = model.predict(input_arr)
        respuesta = np.argmax(predictions)
        if respuesta == 0:
            st.write("Esta imagen corresponde a un 0")   
        elif respuesta==1:
            st.write("Esta imagen corresponde a un 1")
        elif respuesta==2:
            st.write("Esta imagen corresponde a un 2")  
        elif respuesta==3:
            st.write("Esta imagen corresponde a un 3")
        elif respuesta==4:
            st.write("Esta imagen corresponde a un 4")
        elif respuesta==5:
            st.write("Esta imagen corresponde a un 5")
        return respuesta
    predict(file_image)




    



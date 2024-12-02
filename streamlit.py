import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

# Carregar modelo
model = tf.keras.models.load_model('art_style_classifier.h5')
class_names = ['Impressionism', 'Cubism', 'Renaissance', 'Surrealism', 'Dada', 'Expressionism']

st.title("Detecção de Estilos Artísticos em Pinturas")
st.write("Aponte sua câmera para uma pintura para identificar o estilo.")

# Acessar câmera
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if ret:
    # Mostrar quadro
    st.image(frame, channels="BGR")

    # Previsão
    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img, axis=0) / 255.0
    predictions = model.predict(img)
    class_index = np.argmax(predictions)
    confidence = predictions[0][class_index]

    st.write(f"**Estilo Detectado:** {class_names[class_index]}")
    st.write(f"**Confiança:** {confidence*100:.2f}%")
else:
    st.write("Não foi possível acessar a câmera.")
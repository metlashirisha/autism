import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Define image size based on model input shape
IMG_SIZE = (224, 224)  # Change this based on your model's input size

# Streamlit UI
st.title("Autism Facial Recognition Model")
st.write("Upload an image to classify.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Display uploaded image
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess the image
    image = image.resize(IMG_SIZE)  # Resize to match model input
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(image)

    # Display prediction result
    st.write(f"Prediction: {np.argmax(prediction)}")  # Adjust based on your model's output

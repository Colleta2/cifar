import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model('cnn_cifar10.h5')  # Fixed: Corrected function name from `load_model`.

# Model labels
labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

def process_image(image):
    """
    Processes the image for prediction.
    Resizes it to 32x32, normalizes pixel values, and expands dimensions.
    """
    image = image.resize((32, 32))  # Resize to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict(image):
    """
    Predicts the label of the uploaded image using the loaded model.
    """
    processed_image = process_image(image)
    predictions = model.predict(processed_image)
    return labels[np.argmax(predictions)]  # Get label with the highest probability

# Streamlit app
st.title('Image Prediction with CNN (CIFAR-10) Model')
st.write('Upload an image to make a prediction with the trained model.')

# Allow file upload
uploaded_file = st.file_uploader('Choose an image for upload', type=['jpg', 'jpeg', 'png'])  # Fixed: Corrected `type` syntax

if uploaded_file is not None:
    # Open the image file
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    st.write('Predicting...')
    prediction = predict(image)  # Call predict function with the uploaded image
    st.write(f'Prediction: **{prediction}**')

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Define class labels
classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("Model/model3.keras")
    return model

model = load_model()

# Streamlit UI
st.title("Traffic Sign Classifier 🚦")
st.write("Upload a traffic sign image and the model will predict its class.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file).convert('RGB')
    resized = image.resize((30, 30))
    input_data = np.expand_dims(resized, axis=0)
    input_data = np.array(input_data) / 255.0

    # Prediction
    pred = model.predict(input_data)
    result = np.argmax(pred)
    label = classes.get(result, "Unknown")

    # Display results
    st.image(image, caption=f"Uploaded Image", use_column_width=True)
    st.markdown(f"### 🧠 Predicted Class: {result}")
    st.markdown(f"### 🏷️ Label: {label}")

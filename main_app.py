# Library imports
import numpy as np
import streamlit as st
import cv2
from tensorflow.keras.models import load_model
import tensorflow as tf

# Custom CSS styling
st.markdown("""
<style>
    .main {
        background-color: #e6ffe6;
        text-align: center;
        padding: 2rem;
    }
    .stButton>button {
        background-color: #38a169;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        transition: all 0.3s;
        margin: 0 auto;
        display: block;
    }
    .stButton>button:hover {
        background-color: #2f855a;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .upload-text {
        color: #276749;
        font-size: 1.2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-text {
        color: #2f855a;
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #c6f6d5;
        animation: fadeIn 0.5s ease-in;
        margin: 1rem auto;
        max-width: 600px;
        text-align: center;
    }
    .content-section {
        background-color: #f0fff4;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem auto;
        max-width: 800px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .header {
        text-align: center;
        color: #1a472a;
        margin-bottom: 2rem;
    }
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# Loading the Model
model = load_model('plant_disease_model.h5')
                    
# Name of Classes
CLASS_NAMES = ('Tomato-Bacterial_spot', 'Potato-Barly blight', 'Corn-Common_rust')

# Setting Title of App with custom styling
st.markdown("<div class='header'><h1>🌿 Plant Disease Detection </h1></div>", unsafe_allow_html=True)
st.markdown("<div class='content-section'>", unsafe_allow_html=True)
st.markdown("<p class='upload-text'>Upload a clear image of the plant leaf for analysis</p>", unsafe_allow_html=True)

# Creating columns for better layout
col1, col2, col3 = st.columns([1,2,1])

with col2:
    # Uploading the plant image
    plant_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    submit = st.button('🔍 Analyze Leaf')

# On predict button click
if submit:
    if plant_image is not None:
        # Show loading spinner
        with st.spinner('Analyzing your image...'):
            # Convert the file to an opencv image
            file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            
            # Displaying the image with caption
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            st.image(opencv_image, channels="BGR", caption="Uploaded Leaf Image", width=400)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Resizing the image
            opencv_image = cv2.resize(opencv_image, (256, 256))
            
            # Convert image to 4 Dimension
            opencv_image.shape = (1, 256, 256, 3)
            
            # Make Prediction
            Y_pred = model.predict(opencv_image)
            result = CLASS_NAMES[np.argmax(Y_pred)]
            confidence = np.max(Y_pred) * 100
            
            # Display result with animation
            st.markdown(f"""
                <div class='result-text'>
                    <h3>Analysis Results:</h3>
                    <p>Plant Type: {result.split('-')[0]}</p>
                    <p>Condition: {result.split('-')[1]}</p>
                    <hr>
                    <h4>Treatment Recommendations:</h4>
                    <p>• Regular monitoring of plant health</p>
                    <p>• Proper irrigation and drainage</p>
                    <p>• Application of appropriate fungicides if needed</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Additional information
            st.markdown("""
                <div class='content-section'>
                    <h3>Tips for Better Results:</h3>
                    <p>💡 Ensure the leaf is well-lit and centered in the image</p>
                    <p>📸 Use a solid background for better contrast</p>
                    <p>🔍 Take close-up shots of affected areas</p>
                </div>
            """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

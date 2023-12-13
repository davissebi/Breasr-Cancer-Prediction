import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow import keras
import joblib
import pandas as pd  # Import pandas for DataFrame

# Load your trained models
cnn_model = keras.models.load_model('cnn_model.h5')  # Replace with your CNN model file
rf_model = joblib.load('RF_model.joblib')  # Replace with your Random Forest model file

# Define class labels for interpretation
class_labels_cnn = {0: 'Non-Cancer', 1: 'Cancer', 2: 'Malignant'}
class_labels_rf = {0: 'Negative', 1: 'Positive'}

# Function to preprocess a CBIS-DDSM image
def preprocess_cbis_ddsm_image(image):
    # Preprocess the image here (resize, normalization, etc.)
    # Check if the image is grayscale or color
    if len(image.shape) == 2:  # Grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert to RGB
    # Resize the image to (50, 50)
    image = cv2.resize(image, (50, 50), interpolation=cv2.INTER_LINEAR)
    # Normalize pixel values
    image = image / 255.0
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

# Main Streamlit app
st.title("Medical Breast Cancer Classifier")

st.sidebar.title("Select Model")
model_choice = st.sidebar.radio("Choose a Model", ("CNN (Image)", "Random Forest (Details)"))

st.sidebar.title("Upload Image / Data")

if model_choice == "CNN (Image)":
    uploaded_file = st.sidebar.file_uploader("Upload a CBIS-DDSM Image", type=["jpg", "jpeg", "PNG", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Classify"):
            # Preprocess the uploaded image based on type (grayscale or color)
            image_array = np.array(image)
            preprocessed_image = preprocess_cbis_ddsm_image(image_array)

            # Make predictions using the CNN model
            cnn_predictions = cnn_model.predict(preprocessed_image)
            cnn_class = np.argmax(cnn_predictions)
            cnn_label = class_labels_cnn[cnn_class]

            st.success(f"Prediction: {cnn_label}")

else:
    st.sidebar.text("Input histopathological data for Random Forest model.")
    
    # Add input fields for histopathological data
    score = st.sidebar.slider("BI-RADS assessment (Score): 1 to 5 (ordinal, non-predictive!)", 1, 5, 5)
    age = st.sidebar.slider("Age: patient's age in years (integer)", 0, 100, 50)
    shape = st.sidebar.selectbox("Shape: mass shape Round = 1, Oval = 2, Lobular = 3, Irregular = 4 (Nominal)", [1, 2, 3, 4])
    margin = st.sidebar.selectbox("Margin: mass margin circumscribed = 1, microlobulated = 2, obscured = 3, ill-defined = 4, spiculated = 5, (Nominal)", [1, 2, 3, 4, 5])
    density = st.sidebar.selectbox("Density: mass density high = 1, iso = 2, low = 3, fat-containing = 4 (Ordinal)", [1, 2, 3, 4])

    if st.sidebar.button("Predict"):
        # Create a dictionary for input data
        input_data = {
            'Score': [score],
            'Age': [age],
            'Shape': [shape],
            'Margin': [margin],
            'Density': [density]
        }

        # Convert the dictionary to a Pandas DataFrame
        input_data_df = pd.DataFrame(input_data)

        # Make predictions using the Random Forest model
        rf_predictions = rf_model.predict(input_data_df)

        # Interpret the predictions
        rf_class = rf_predictions[0]  # Assuming only one prediction is made
        rf_label = class_labels_rf[rf_class]

        st.success(f"Prediction: {rf_label}")

st.write("")



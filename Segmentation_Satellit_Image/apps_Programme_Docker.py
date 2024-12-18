import streamlit as st
import cv2
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
from PIL import Image
import segmentation_models as sm
import smooth_tiled_predictions
from sklearn.preprocessing import MinMaxScaler
from smooth_tiled_predictions import predict_img_with_smooth_windowing
#from keras.models import load_model
import segmentation_models as sm
import tensorflow as tf




import base64
import plotly.express as px

# Set page layout
st.set_page_config(layout="wide")


df = px.data.iris()

#@st.experimental_memo
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("dom.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://raw.githubusercontent.com/MarDom15/ML-Python/main/intelligence-artificielle-puce-ia-future-innovation-technologique_53876-129780.webp");
background-size: 180%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/jpg;base64,{img}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)



# Define label colors dictionary (matching your specifications)
label_colors = {
    "Unlabeled": (128, 128, 128),  # Gray
    "Freespace": (0, 255, 0),  # Light green
    "Water": (0, 0, 255),  # Blue
    "Building": (255, 0, 0),  # Red
    "Tree": (0, 128, 0)  # Dark green
}

# Define function to convert labels to colored image
def label_to_rgb(predicted_image):
    class_colors = {
        0: (128, 128, 128),  # Gray (Unlabeled)
        1: (0, 255, 0),  # Green (Freespace)
        2: (0, 0, 255),  # Blue (Water)
        3: (255, 0, 0),  # Red (Building) 
        4: (0, 128, 0)   # Dark Green (Tree)
    }

    colored_image = np.zeros((predicted_image.shape[0], predicted_image.shape[1], 3), dtype=np.uint8)

    for class_id, color in class_colors.items():
        colored_image[predicted_image == class_id] = color

    return colored_image

# Create a layout with a sidebar
#st.set_page_config(layout="wide")

col1, col2 = st.columns([1, 4])  # Adjust column widths as needed

# Display color legend in the sidebar
with col1:
    st.subheader("Color Legend")
    for label, color in label_colors.items():
        colored_square = np.full((30, 60, 3), color, dtype=np.uint8)  # Create a colored square
        st.write(f"- ", end="")
        st.image(colored_square)  # Display colored square
        st.write(f"{label}")

# Chemin local vers l'image de fond
background_image_path = '/app/dom.jpg'

# Charger l'image de fond
background_image = Image.open(background_image_path)

# Appliquer une astuce CSS pour définir l'image de fond sur toute la page
st.markdown(
    f"""
    <style>
        .reportview-container {{
            background: url(data:image/jpg;base64,{background_image}) center;
            background-size: cover;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Main app content
with col2:
    st.title("Image Segmentation ")
    backbone_options = ['resnet18', 'resnet34', 'resnet50', 'resnet101']
    selected_backbone = st.selectbox("Select Backbone", backbone_options)

    model_options = {
        'resnet18': ["Resnet18_e100.hdf5", "Resnet18_e25.hdf5"],
        'resnet34': ["Resnet34_e100.hdf5", "Resnet34_e25.hdf5", "Resnet34_e300.hdf5"],
        'resnet50': ["Resnet50_e100.hdf5", "Resnet50_e25.hdf5", "Resnet50_e300.hdf5"],
        'resnet101': ["Resnet101_e100.hdf5", "Resnet101_e25.hdf5", "Resnet101_e300.hdf5"]
    }
    selected_model_names = model_options[selected_backbone]
    selected_model_name = st.selectbox("Select Model", selected_model_names)
    selected_model_path = "/app/" + selected_model_name
    selected_model = tf.keras.models.load_model(selected_model_path, compile=False)  # Move this line here
    
    # Charger le modèle à partir du fichier HDF5 en utilisant la méthode sm.load
    #selected_model = tf.keras.models.load_model(selected_model_path, compile=False)
    
    preprocess_input = sm.get_preprocessing(selected_backbone)  # Define preprocess_input based on selected backbone
    n_classes = 5
    patch_size = 256  # Define patch size
    scaler = MinMaxScaler()  # Define scaler

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "tif", "tiff"])

    if uploaded_file is not None:
        img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

        input_img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
        input_img = preprocess_input(input_img)

        st.image(img, caption="Original Image")

        # Choice for comparison
        if st.checkbox("Compare two models"):
            second_backbone = st.selectbox("Select Second Backbone", backbone_options)
            second_selected_model_names = model_options[second_backbone]
            second_selected_model_name = st.selectbox("Select Second Model", second_selected_model_names)
            second_selected_model_path = "/app/" + second_selected_model_name
            second_model = tf.keras.models.load_model(second_selected_model_path, compile=False)

            if input_img is not None:
                with st.spinner("Generating predictions..."):
                    predictions_smooth_1 = predict_img_with_smooth_windowing(
                        input_img,
                        window_size=patch_size,
                        subdivisions=2,
                        nb_classes=n_classes,
                        pred_func=lambda img_batch_subdiv: selected_model.predict(img_batch_subdiv)
                    )
                    final_prediction_1 = np.argmax(predictions_smooth_1, axis=2)

                    predictions_smooth_2 = predict_img_with_smooth_windowing(
                        input_img,
                        window_size=patch_size,
                        subdivisions=2,
                        nb_classes=n_classes,
                        pred_func=lambda img_batch_subdiv: second_model.predict(img_batch_subdiv)
                    )
                    final_prediction_2 = np.argmax(predictions_smooth_2, axis=2)

                st.image(label_to_rgb(final_prediction_1),
                         caption=f"Predicted Segmentation Mask - {selected_backbone} Model: {selected_model_name}")
                st.image(label_to_rgb(final_prediction_2),
                         caption=f"Predicted Segmentation Mask - {second_backbone} Model: {second_selected_model_name}")

        else:
            if input_img is not None:
                with st.spinner("Generating prediction..."):
                    predictions_smooth = predict_img_with_smooth_windowing(
                        input_img,
                        window_size=patch_size,
                        subdivisions=2,
                        nb_classes=n_classes,
                        pred_func=lambda img_batch_subdiv: selected_model.predict(img_batch_subdiv)
                    )
                    final_prediction = np.argmax(predictions_smooth, axis=2)

                st.image(label_to_rgb(final_prediction),
                         caption=f"Predicted Segmentation Mask - {selected_backbone} Model: {selected_model_name}")
            else:
                st.write("Please upload an image to start.")

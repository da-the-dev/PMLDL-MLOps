import os
from PIL import Image
import streamlit as st

from src.util.ml import inference, load_model
from src.datasets.labels import course_classes

MODEL = "effnet-b2_epoch_10.pkl"
if os.getenv("PROD"):
    MODEL = "model.pkl"
    
model = load_model(MODEL)

st.text("Fixed width text")

uploaded_image = st.file_uploader(
    "Choose an to classify...",
    type=["jpg", "jpeg", "png"],
)

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Button to run inference
    if st.button("Run Inference"):
        st.write("Running inference...")

        # Run inference on the image
        result = inference(
            model,
            image,
            course_classes,
        )

        # Display the result
        st.write(f"Inference Result: {result}")

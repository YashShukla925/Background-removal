import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from isnet import ISNet  # Ensure you have ISNet model definition

# Load the IS-Net model
@st.cache_resource
def load_model():
    model = ISNet()  # Initialize the model
    model.load_state_dict(torch.load("isnet.pth", map_location=torch.device("cpu")))  # Load weights
    model.eval()  # Set model to evaluation mode
    return model

# Function to remove background
def remove_background(image, model):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512))
    ])

    image_tensor = transform(image).unsqueeze(0)  # Convert image to tensor

    with torch.no_grad():
        mask = model(image_tensor)  # Get background mask

    mask = mask.squeeze().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255  # Convert to binary mask

    image_np = np.array(image)
    image_np = cv2.resize(image_np, (512, 512))  # Resize to match mask
    result = cv2.bitwise_and(image_np, image_np, mask=mask)  # Apply mask

    return result

# Streamlit UI
st.title("Background Removal using IS-Net")
st.write("Upload an image to remove the background.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    st.write("Processing...")
    model = load_model()
    result = remove_background(image, model)
    result_image = Image.fromarray(result)
    
    st.image(result_image, caption="Background Removed", use_column_width=True)
    
    st.download_button("Download Processed Image", data=result_image.tobytes(), file_name="processed.png", mime="image/png")

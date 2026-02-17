import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from huggingface_hub import login

st.set_page_config(page_title="Image to Caption", page_icon="📸")

st.title("📸 Image Caption Generator")

# Get token from Streamlit Secrets
HF_TOKEN = st.secrets["HF_TOKEN"]

# Login to Hugging Face
login(token=HF_TOKEN)

device = torch.device("cpu")

@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model.to(device)
    model.eval()
    return processor, model

processor, model = load_model()

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width="stretch")

    if st.button("Generate Caption"):
        with torch.no_grad():
            inputs = processor(image, return_tensors="pt").to(device)
            output = model.generate(**inputs, max_length=50, num_beams=5)
            caption = processor.decode(output[0], skip_special_tokens=True)

        st.success("Generated Caption:")
        st.write(caption)


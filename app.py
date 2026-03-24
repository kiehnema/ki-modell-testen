import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

st.title("🌿 Pflanzenkrankheiten Erkennung mit KI")

@st.cache_resource
def load_model():
    model_name = "Daksh159/plant-disease-mobilenetv2"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    return processor, model

processor, model = load_model()

def clean_label(label):
    return label.replace("___", " - ").replace("_", " ")

uploaded_file = st.file_uploader("📷 Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)

    with st.spinner("🔍 Analysiere..."):
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        probs = torch.nn.functional.softmax(logits, dim=1)[0]
        top_k = torch.topk(probs, 3)

    st.subheader("📊 Ergebnis:")

    for i in range(3):
        idx = top_k.indices[i].item()
        score = top_k.values[i].item() * 100
        label = model.config.id2label[idx]

        st.write(f"**{clean_label(label)}** → {score:.2f}%")
        st.progress(score / 100)

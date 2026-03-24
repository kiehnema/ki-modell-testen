import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Titel der App
st.title("🌿 Pflanzenkrankheiten Klassifikation")
st.write("Lade ein Bild einer Pflanze hoch, um die Krankheit zu erkennen.")

# Modell laden (einmalig)
@st.cache_resource
def load_model():
    model_name = "Mamdouh-Alaa12/Plants_Disease_Classification"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    return processor, model

processor, model = load_model()

# Bild-Upload
uploaded_file = st.file_uploader("📷 Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Bild anzeigen
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    # Vorhersage
    with st.spinner("🔍 Analysiere Bild..."):
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        predicted_class_id = logits.argmax(-1).item()
        label = model.config.id2label[predicted_class_id]

    # Ergebnis anzeigen
    st.success(f"🌱 Vorhersage: **{label}**")

import streamlit as st
from transformers import pipeline
from PIL import Image

# Titel
st.title("🌿 Pflanzenkrankheiten Erkennung mit KI")
st.write("Lade ein Bild einer Pflanze hoch, um mögliche Krankheiten zu erkennen.")

# Modell laden (wird gecached)
@st.cache_resource
def load_model():
    model = pipeline(
        "image-classification",
        model="Daksh159/plant-disease-mobilenetv2"
    )
    return model

classifier = load_model()

# Labels bereinigen
def clean_label(label):
    return label.replace("___", " - ").replace("_", " ")

# Datei-Upload
uploaded_file = st.file_uploader("📷 Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    with st.spinner("🔍 Analysiere Bild..."):
        results = classifier(image)

    st.subheader("📊 Ergebnis (Top 3):")

    for r in results[:3]:
        label = clean_label(r["label"])
        score = r["score"] * 100
        st.write(f"**{label}** → {score:.2f}%")
        st.progress(min(score / 100, 1.0))

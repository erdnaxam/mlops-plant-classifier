"""
Interface Gradio pour tester le modèle de classification.
"""
import os
import requests
import gradio as gr
from PIL import Image
import io

API_URL = os.getenv("API_URL", "http://localhost:8000")


def classify_image(image: Image.Image):
    """Envoie l'image à l'API et retourne la prédiction."""
    if image is None:
        return "Veuillez uploader une image.", {}

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    try:
        response = requests.post(
            f"{API_URL}/predict",
            files={"file": ("image.jpg", buffer, "image/jpeg")},
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()

        label = result["prediction"].upper()
        confidence = result["confidence"] * 100
        probs = result["probabilities"]

        summary = f"**{label}** ({confidence:.1f}% de confiance)"
        return summary, probs

    except requests.exceptions.ConnectionError:
        return "❌ API non disponible. Vérifiez que le service est lancé.", {}
    except Exception as e:
        return f"❌ Erreur : {str(e)}", {}


demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Uploadez une image (dandelion ou grass)"),
    outputs=[
        gr.Markdown(label="Résultat"),
        gr.Label(label="Probabilités par classe"),
    ],
    title="🌿 Plant Image Classifier",
    description=(
        "Modèle de deep learning (ResNet18 via FastAI) entraîné à distinguer "
        "les **pissenlits (dandelion)** de l'**herbe (grass)**.\n\n"
        "Uploadez une image pour obtenir une prédiction."
    ),
    examples=[],
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

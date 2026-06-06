"""
webapp/app.py
Interface Gradio pour tester le modèle de classification d'images.
"""

import os
import requests
import gradio as gr
from PIL import Image
import io

API_URL = os.getenv("API_URL", "http://localhost:8000")


def predict_image(image: Image.Image) -> tuple[str, dict]:
    """Envoie une image à l'API et retourne la prédiction."""
    if image is None:
        return "Aucune image fournie", {}

    try:
        # Convertir en bytes
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        buf.seek(0)

        response = requests.post(
            f"{API_URL}/predict",
            files={"file": ("image.jpg", buf, "image/jpeg")},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        label = data["label"].upper()
        confidence = data["confidence"]
        probs = data["probabilities"]

        result = f"{'🌼' if label == 'DANDELION' else '🌿'} **{label}** — confiance : {confidence:.1%}"
        return result, probs

    except requests.exceptions.ConnectionError:
        return "❌ API non disponible. Vérifiez que l'API est lancée.", {}
    except Exception as e:
        return f"❌ Erreur: {str(e)}", {}


def check_api_health() -> str:
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        data = r.json()
        if data["model_loaded"]:
            return "✅ API connectée — Modèle chargé"
        return "⚠️ API connectée mais modèle non chargé"
    except:
        return "❌ API non disponible"


# ─── Interface Gradio ────────────────────────────────────────────────
with gr.Blocks(title="🌿 Plant Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌿 Plant Classifier\nClassification d'images : **Pissenlit** ou **Herbe** ?")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="📷 Image à classifier")
            predict_btn = gr.Button("🔍 Classifier", variant="primary")
            health_btn = gr.Button("💊 Vérifier l'API", variant="secondary")

        with gr.Column():
            result_output = gr.Markdown(label="Résultat")
            prob_output = gr.Label(label="Probabilités par classe", num_top_classes=2)
            health_output = gr.Textbox(label="Statut de l'API", interactive=False)

    predict_btn.click(
        fn=predict_image,
        inputs=[image_input],
        outputs=[result_output, prob_output],
    )

    health_btn.click(
        fn=check_api_health,
        outputs=[health_output],
    )

    gr.Examples(
        examples=[],
        inputs=[image_input],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

"""
=================================================================================
 SMART GREENHOUSE MONITORING SYSTEM - AI COMPUTE SERVER
=================================================================================
 Framework:  Flask
 Purpose:    Receives a JPEG frame + sensor telemetry from the ESP32-CAM edge
             device via multipart/form-data POST, preprocesses the image to
             224x224 for a MobileNetV2-style CNN input, runs a (mocked)
             foliage-disease classification inference, and returns a strict
             JSON payload: {"status", "class", "health_score"}.

 Deployment: Designed to run on Render.com free-tier web services.
   - Start Command:  gunicorn app:app
   - Build Command:  pip install -r requirements.txt
   - Python version: 3.10+ recommended

 To swap the mock stub for a real TensorFlow Lite MobileNetV2 model, replace
 the body of `run_inference()` with an actual `tflite_runtime.Interpreter`
 call (see comments inside that function for the exact integration points).
=================================================================================
"""

import io
import logging
import random

from flask import Flask, request, jsonify
from PIL import Image, UnidentifiedImageError
import numpy as np

# ---------------------------------------------------------------------------
# App & logging setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("greenhouse-server")

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
MODEL_INPUT_SIZE = (224, 224)   # Standard MobileNetV2 input footprint
DISEASE_CLASSES = [
    "Healthy_Leaf_Tissue",
    "Early_Blight",
    "Late_Blight",
    "Powdery_Mildew",
    "Bacterial_Spot",
]


# =============================================================================
#  IMAGE PRE-PROCESSING PIPELINE
# =============================================================================
def preprocess_image(file_stream):
    """
    Reads a raw JPEG byte stream, decodes it with Pillow, resizes to the
    model's expected 224x224 input, converts to RGB, and normalizes pixel
    values to the [0, 1] float range expected by most MobileNetV2 variants.

    Raises:
        ValueError: if the stream is empty, corrupt, or not a valid image.

    Returns:
        np.ndarray of shape (224, 224, 3), dtype float32, values in [0, 1].
    """
    raw_bytes = file_stream.read()
    if not raw_bytes or len(raw_bytes) == 0:
        raise ValueError("Empty image payload received.")

    try:
        image = Image.open(io.BytesIO(raw_bytes))
        image.load()  # Force decode now, so truncated/corrupt JPEGs raise here.
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Corrupt or unreadable image data: {exc}")

    # Ensure 3-channel RGB (handles grayscale / RGBA / CMYK edge cases)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize/scale to the CNN's expected footprint
    image = image.resize(MODEL_INPUT_SIZE, Image.BILINEAR)

    # Convert to normalized float32 numpy array
    image_array = np.asarray(image, dtype=np.float32) / 255.0

    if image_array.shape != (MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1], 3):
        raise ValueError(f"Unexpected image array shape: {image_array.shape}")

    return image_array


# =============================================================================
#  DEEP LEARNING INFERENCE STUB (mock TensorFlow Lite MobileNetV2)
# =============================================================================
def run_inference(image_array):
    """
    Mock stand-in for a pre-trained TensorFlow Lite MobileNetV2 classifier
    that would detect foliage disease (Blight, Mildew, etc.) in a leaf image.

    --- REAL MODEL INTEGRATION POINT ---
    To use an actual TFLite model, replace this function body with:

        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path="mobilenetv2_foliage.tflite")
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        batched = np.expand_dims(image_array, axis=0).astype(input_details[0]["dtype"])
        interpreter.set_tensor(input_details[0]["index"], batched)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]["index"])[0]

        class_index = int(np.argmax(predictions))
        confidence = float(predictions[class_index])
        predicted_class = DISEASE_CLASSES[class_index]
        health_score = confidence if predicted_class == "Healthy_Leaf_Tissue" else (1.0 - confidence)
        return predicted_class, health_score

    For now, this stub derives a pseudo-deterministic "confidence" from basic
    image statistics (mean green-channel intensity) so results are at least
    stable/repeatable for a given image, then layers in a small random jitter
    to simulate model uncertainty.
    """
    # Simple heuristic seed: greener, brighter leaves -> nudged toward healthy.
    green_channel_mean = float(np.mean(image_array[:, :, 1]))
    overall_brightness = float(np.mean(image_array))

    heuristic_health_bias = np.clip((green_channel_mean * 0.6) + (overall_brightness * 0.4), 0.0, 1.0)

    # Simulate class probabilities using the heuristic as a soft prior
    if heuristic_health_bias > 0.55:
        predicted_class = "Healthy_Leaf_Tissue"
        confidence = round(min(0.99, heuristic_health_bias + random.uniform(0.0, 0.15)), 3)
        health_score = confidence
    else:
        predicted_class = random.choice(DISEASE_CLASSES[1:])
        confidence = round(min(0.99, (1.0 - heuristic_health_bias) + random.uniform(0.0, 0.15)), 3)
        health_score = round(1.0 - confidence, 3)

    health_score = float(np.clip(health_score, 0.0, 1.0))

    logger.info("Inference stub result: class=%s, health_score=%.3f", predicted_class, health_score)
    return predicted_class, health_score


# =============================================================================
#  /predict ENDPOINT
# =============================================================================
@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts multipart/form-data containing:
      - "image": raw JPEG binary file part (required)
      - "temperature", "humidity", "soil_moisture", "lux": optional text fields

    Returns JSON: {"status": "success", "class": <str>, "health_score": <float>}
    or {"status": "error", "message": <str>} with an appropriate HTTP code.
    """
    try:
        if "image" not in request.files:
            logger.warning("Request rejected: no 'image' part in multipart payload.")
            return jsonify({"status": "error", "message": "Missing 'image' file part."}), 400

        image_file = request.files["image"]

        # Log any accompanying telemetry fields for observability (optional).
        telemetry_snapshot = {
            "temperature": request.form.get("temperature"),
            "humidity": request.form.get("humidity"),
            "soil_moisture": request.form.get("soil_moisture"),
            "lux": request.form.get("lux"),
        }
        logger.info("Received telemetry snapshot: %s", telemetry_snapshot)

        try:
            image_array = preprocess_image(image_file.stream)
        except ValueError as ve:
            logger.warning("Image preprocessing failed: %s", ve)
            return jsonify({"status": "error", "message": str(ve)}), 400

        predicted_class, health_score = run_inference(image_array)

        response_payload = {
            "status": "success",
            "class": predicted_class,
            "health_score": round(float(health_score), 3),
        }
        return jsonify(response_payload), 200

    except Exception as exc:  # noqa: BLE001 - top-level safety net for edge-device reliability
        logger.exception("Unhandled exception in /predict")
        return jsonify({"status": "error", "message": f"Internal server error: {exc}"}), 500


@app.route("/", methods=["GET"])
def health_check():
    """Simple liveness endpoint so Render's health checks don't fail."""
    return jsonify({"status": "ok", "service": "greenhouse-ai-compute-server"}), 200


if __name__ == "__main__":
    # Local dev only. On Render, gunicorn serves the `app` object directly.
    app.run(host="0.0.0.0", port=5000, debug=False)

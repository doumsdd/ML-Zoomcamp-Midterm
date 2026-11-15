"""
Pokemon Type1 Predictor - Flask Web API

POST /predict {"name": "bulbasaur"} -> {"prediction": "Grass", "confidence": 0.98}
GET /health -> {"status": "ok"}
"""

import os
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model at startup
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / 'model.pkl'
FEATURE_NAMES_PATH = BASE_DIR / 'feature_names.json'
CLASS_LABELS_PATH = BASE_DIR / 'class_labels.json'


def load_model():
    """Load the trained model and feature names."""
    with open(MODEL_PATH, 'rb') as f:
        model = joblib.load(f)
    with open(FEATURE_NAMES_PATH, 'r') as f:
        feature_names = json.load(f)
    
    # Load class labels if available (for XGBoost-encoded models)
    class_labels = None
    if CLASS_LABELS_PATH.exists():
        with open(CLASS_LABELS_PATH, 'r') as f:
            class_labels = json.load(f)
    
    return model, feature_names, class_labels


# Global model variable
model, FEATURE_NAMES, CLASS_LABELS = load_model()


def preprocess_name(name):
    """Convert a name string to feature vector matching training format."""
    name = str(name).lower().strip()
    vowels = set('aeiou')

    import re
    features = {
        'name_length': len(name),
        'vowel_ratio': sum(1 for c in name if c in vowels) / max(len(name), 1),
        'consonant_ratio': sum(1 for c in name if c.isalpha() and c not in vowels) / max(len(name), 1),
        'ends_with_vowel': int(name[-1] in vowels if name else 0),
        'has_double_letter': int(bool(re.search(r'([a-z])\1', name))),
        'unique_letter_ratio': len(set(name)) / max(len(name), 1),
        'first_letter': name[0] if name else ' ',
        'contains_digit': int(any(c.isdigit() for c in name)),
        'syllable_estimate': max(1, sum(1 for i in range(len(name))
                    if name[i] in vowels
                    and (i == 0 or name[i-1] not in vowels)
                    and (i == len(name)-1 or name[i+1] not in vowels))),
        'name_length_squared': len(name) ** 2,
    }
    return pd.DataFrame([features])[FEATURE_NAMES]


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


@app.route('/predict', methods=['POST'])
def predict():
    """Predict Pokemon type from name."""
    data = request.get_json()

    if not data or 'name' not in data:
        return jsonify({"error": "Missing 'name' field. Example: {'name': 'bulbasaur'}"}), 400

    name = data['name']
    if not name or len(name.strip()) == 0:
        return jsonify({"error": "Name cannot be empty"}), 400

    try:
        # Preprocess and predict
        X = preprocess_name(name)
        pred_int_raw = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]

        # Convert numpy int to Python int for proper indexing
        pred_int = int(pred_int_raw) if isinstance(pred_int_raw, (int, np.integer)) else pred_int_raw

        # Decode prediction if we have class labels
        if CLASS_LABELS is not None:
            classes = CLASS_LABELS
        else:
            classes = model.classes_

        confidence = float(probabilities[pred_int]) if isinstance(pred_int, int) else 0.5
        prediction = classes[pred_int] if isinstance(pred_int, int) else str(pred_int)

        result = {
            "name": name,
            "prediction": prediction,
            "confidence": round(confidence, 4),
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 9696))
    app.run(host='0.0.0.0', port=port, debug=False)

# Pokemon Type1 Predictor

## Problem Description

This project builds a **multi-class classification model** that predicts the primary type (`Type1`) of a Pokemon based solely on its **name**. Given 803 Pokemon names from Gen 1-9, the model learns patterns in naming conventions associated with each of the 18 Pokemon types (e.g., Grass, Fire, Water, Electric, etc.).

### Why is this interesting?
- Pokemon names often contain linguistic clues related to their type (e.g., "fire" sounds for Fire-types, water-related words for Water-types)
- This demonstrates how even simple text-derived features can capture meaningful semantic information
- It's a realistic multi-class problem with 18 imbalanced classes

### Use Cases
- Fun exploration of Pokemon naming patterns
- Demonstrates NLP feature extraction without using embeddings or transformers
- Shows end-to-end ML pipeline: EDA -> preprocessing -> tuning -> deployment

## Dataset

- **Source**: `data/pokemon.csv` included in repo
- **Samples**: ~803 Pokemon (Gen 1 - Gen 9)
- **Features used**: Only `Name` column is input; `Type1` is target
- **Classes**: 18 Pokemon types

## Project Structure

```
pokemon-type-predictor/
├── data/
│   └── pokemon.csv              # Raw dataset
├── notebook.ipynb               # EDA + Feature Engineering + Benchmarking
├── pipeline.py                  # sklearn pipeline + feature engineering
├── train.py                     # Training script
├── predict.py                   # Flask web API
├── model.pkl                    # Trained model (generated)
├── feature_names.json           # Feature list (generated)
├── training_metrics.json        # Evaluation metrics (generated)
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container definition
├── README.md                    # This file
└── tests/
    └── test_pipeline.py         # Unit tests
```

## Quick Start

### 1. Setup

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train.py
```

This will:
- Load and preprocess the data
- Train Logistic Regression, Random Forest, and XGBoost with hyperparameter tuning
- Select the best model by CV F1-macro score
- Save to `model.pkl`

### 3. Test Prediction Locally

```bash
python predict.py
# In another terminal:
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"name": "bulbasaur"}'
```

Expected response:
```json
{
  "name": "bulbasaur",
  "prediction": "Grass",
  "confidence": 0.95,
  "all_probabilities": {
    "Grass": 0.95,
    "Bug": 0.03,
    ...
  }
}
```

### 4. Run with Docker

```bash
docker build -t pokemon-predictor .
docker run -p 9696:9696 pokemon-predictor
curl -X POST http://localhost:9696/predict -d '{"name":"charizard"}'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Predict Pokemon type |
| `/health` | GET | Health check |

### Predict Request Body
```json
{"name": "squirtle"}
```

### Predict Response
```json
{
  "name": "squirtle",
  "prediction": "Water",
  "confidence": 0.87,
  "all_probabilities": {"Water": 0.87, "Normal": 0.05, ...}
}
```

## EDA Highlights

See `notebook.ipynb` for comprehensive analysis including:
- Distribution of Pokemon types
- Name length statistics per type
- Feature importance from trained models
- Correlation analysis
- Misclassification patterns

## Models Compared

| Model | Expected Accuracy | Notes |
|-------|------------------|-------|
| Logistic Regression | ~55-65% | Baseline, fast training |
| Random Forest | ~65-75% | Captures non-linear patterns |
| XGBoost | ~68-78% | Best overall performance |

## Technology Stack

- **ML**: scikit-learn, XGBoost
- **Web**: Flask + Gunicorn
- **Container**: Docker
- **Data**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

## How Features Are Derived from Names

| Feature | Description |
|---------|-------------|
| name_length | Total characters in name |
| vowel_ratio | Proportion of vowels |
| consonant_ratio | Proportion of consonants |
| ends_with_vowel | Boolean: last letter is vowel |
| has_double_letter | Boolean: repeated consecutive letters |
| unique_letter_ratio | Diversity of letters used |
| first_letter | Starting character (one-hot encoded) |
| contains_digit | Boolean: contains any digit |
| syllable_estimate | Estimated syllables via vowel clusters |
| name_length_squared | Non-linear length effect |

## License

MIT - Feel free to use and modify!

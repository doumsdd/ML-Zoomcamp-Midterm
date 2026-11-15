#!/usr/bin/env python3
"""Quick train - train models and save best to model.pkl."""
import json
import pickle
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from pipeline import load_and_preprocess, get_feature_names

NUMERIC_FEATS = ['name_length', 'vowel_ratio', 'consonant_ratio', 'ends_with_vowel',
                 'has_double_letter', 'unique_letter_ratio', 'contains_digit',
                 'syllable_estimate', 'name_length_squared']
CAT_FEATS = ['first_letter']

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).parent


def build_pipe(clf):
    return Pipeline([
        ('preprocessor', ColumnTransformer([
            ('numeric', StandardScaler(), NUMERIC_FEATS),
            ('categorical', OneHotEncoder(handle_unknown='ignore'), CAT_FEATS)
        ])),
        ('classifier', clf)
    ])


def main():
    logger.info("Loading data...")
    X, y_raw = load_and_preprocess()
    cat_y = pd.Categorical(y_raw)
    y_int = cat_y.codes
    categories = cat_y.categories.tolist()
    n_classes = len(categories)
    logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {n_classes} classes")

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=2, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
    }
    try:
        from xgboost import XGBClassifier
        models['XGBoost'] = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                           eval_metric='mlogloss', random_state=42, n_jobs=-1, verbosity=0)
    except ImportError:
        pass

    results = {}
    best_name, best_f1, best_pipe = None, 0, None

    for name, clf in models.items():
        logger.info("--- Training: %s ---", name)
        pipe = build_pipe(clf)
        pipe.fit(X, y_int)
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        cv_acc = cross_val_score(pipe, X, y_int, cv=cv, scoring='accuracy')
        cv_f1 = cross_val_score(pipe, X, y_int, cv=cv, scoring='f1_macro')
        y_pred = pipe.predict(X)
        acc = accuracy_score(y_int, y_pred)
        f1 = f1_score(y_int, y_pred, average='macro')
        results[name] = dict(cv_accuracy_mean=float(cv_acc.mean()), cv_accuracy_std=float(cv_acc.std()),
                             cv_f1_macro_mean=float(cv_f1.mean()), train_accuracy=float(acc), train_f1_macro=float(f1))
        logger.info("CV Acc: %.4f +/- %.4f | CV F1: %.4f | Train Acc: %.4f, F1: %.4f",
                     cv_acc.mean(), cv_acc.std()*2, cv_f1.mean(), acc, f1)
        if cv_f1.mean() > best_f1:
            best_f1 = cv_f1.mean()
            best_name = name
            best_pipe = pipe

    logger.info("*** BEST MODEL: %s (CV F1: %.4f) ***", best_name, best_f1)
    joblib.dump(best_pipe, str(BASE_DIR / 'model.pkl'))
    with open(str(BASE_DIR / 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(categories, f)
    with open(str(BASE_DIR / 'class_labels.json'), 'w') as f:
        json.dump(categories, f)
    with open(str(BASE_DIR / 'feature_names.json'), 'w') as f:
        json.dump(get_feature_names(), f, indent=2)
    metrics = dict(best_model=best_name, total_samples=int(X.shape[0]), num_features=int(X.shape[1]),
                   num_classes=n_classes, models_compared=results, class_labels=categories)
    with open(str(BASE_DIR / 'training_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print("\n=== COMPARISON ===")
    print(f"{'Model':<25} {'CV Acc':>10} {'CV F1':>10}")
    for n, m in results.items():
        marker = " <-- BEST" if n == best_name else ""
        print(f"{n:<25} {m['cv_accuracy_mean']:>10.4f} {m['cv_f1_macro_mean']:>10.4f}{marker}")
    return best_pipe


if __name__ == '__main__':
    main()

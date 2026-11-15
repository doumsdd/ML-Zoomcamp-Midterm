"""
Pokemon Type1 Predictor - sklearn Pipeline Module

Extracts features from Pokemon names and predicts their primary type (Type1).
Uses a scikit-learn Pipeline with preprocessing + classifier.
"""

import re
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import joblib


def extract_features(df):
    """Extract numeric and categorical features from raw dataframe."""
    df = df.copy()
    df['Name'] = df['Name'].astype(str).str.lower().str.strip()

    vowels = set('aeiou')

    # Name length
    df['name_length'] = df['Name'].apply(len)

    # Vowel ratio in name
    def vowel_ratio(name):
        count = sum(1 for c in str(name) if c in vowels)
        return count / max(len(str(name)), 1)
    df['vowel_ratio'] = df['Name'].apply(vowel_ratio)

    # Consonant ratio
    def consonant_ratio(name):
        count = sum(1 for c in str(name) if c.isalpha() and c not in vowels)
        return count / max(len(str(name)), 1)
    df['consonant_ratio'] = df['Name'].apply(consonant_ratio)

    # Ends with vowel
    df['ends_with_vowel'] = df['Name'].apply(lambda x: int(x[-1] in vowels))

    # Contains double letter
    df['has_double_letter'] = df['Name'].apply(
        lambda x: int(bool(re.search(r'([a-z])\1', x)))
    )

    # Unique letters ratio
    def unique_ratio(name):
        return len(set(str(name))) / max(len(str(name)), 1)
    df['unique_letter_ratio'] = df['Name'].apply(unique_ratio)

    # First letter
    first_letter = df['Name'].str[0]
    df['first_letter'] = first_letter

    # Has digit in name
    df['contains_digit'] = df['Name'].apply(
        lambda x: int(any(c.isdigit() for c in x))
    )

    # Syllable estimate (based on vowel clusters)
    def estimate_syllables(name):
        name = str(name)
        vc = sum(1 for i in range(len(name))
                 if name[i] in vowels
                 and (i == 0 or name[i-1] not in vowels)
                 and (i == len(name)-1 or name[i+1] not in vowels))
        return max(1, vc)
    df['syllable_estimate'] = df['Name'].apply(estimate_syllables)

    # Name length squared
    df['name_length_squared'] = df['name_length'] ** 2

    return df


def get_feature_names():
    """Return list of feature names used by the pipeline."""
    return [
        'name_length', 'vowel_ratio', 'consonant_ratio',
        'ends_with_vowel', 'has_double_letter', 'unique_letter_ratio',
        'first_letter', 'contains_digit', 'syllable_estimate', 'name_length_squared'
    ]


def build_pipeline(classifier_type='random_forest'):
    """Build an sklearn pipeline with preprocessing and classifier."""
    numeric_features = ['name_length', 'vowel_ratio', 'consonant_ratio',
                        'ends_with_vowel', 'has_double_letter',
                        'unique_letter_ratio', 'contains_digit',
                        'syllable_estimate', 'name_length_squared']
    categorical_features = ['first_letter']

    preprocessor = ColumnTransformer(transformers=[
        ('numeric', StandardScaler(), numeric_features),
        ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    classifiers = {
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'random_forest': RandomForestClassifier(random_state=42),
        'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss',
                                 random_state=42, n_jobs=-1, verbosity=0),
        'gradient_boosting': GradientBoostingClassifier(random_state=42)
    }

    if classifier_type not in classifiers:
        raise ValueError(f"Unknown classifier_type: {classifier_type}")

    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifiers[classifier_type])
    ])


def get_gridsearch_params(classifier_type='random_forest'):
    """Return hyperparameter grid for GridSearchCV."""
    param_grids = {
        'logistic_regression': {
            'classifier__C': [0.01, 0.1, 1, 10, 100],
            'classifier__penalty': ['l2'],
            'classifier__solver': ['lbfgs'],
        },
        'random_forest': {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2],
        },
        'xgboost': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__subsample': [0.8, 1.0],
        },
        'gradient_boosting': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
        },
    }
    return param_grids.get(classifier_type)


def load_and_preprocess(data_path='data/pokemon.csv'):
    """Load CSV and apply feature engineering. Returns (X, y)."""
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['Type1'])
    df = df[df['Type1'] != '']
    df['Type1'] = df['Type1'].str.strip().str.lower().str.capitalize()
    df = df[['Name', 'Type1']].drop_duplicates()
    df = extract_features(df)
    X = df[get_feature_names()]
    y = df['Type1']
    return X, y


if __name__ == '__main__':
    X, y = load_and_preprocess()
    print(f"Dataset shape: X={X.shape}, y has {y.nunique()} unique types")
    print(f"Classes: {sorted(y.unique())}")

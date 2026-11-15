"""Unit tests for pipeline.py feature engineering and loading."""

import pytest
import pandas as pd
from pipeline import extract_features, load_and_preprocess, get_feature_names


def test_extract_features_produces_expected_columns():
    """Test that extract_features creates all expected columns."""
    df = pd.DataFrame({'Name': ['bulbasaur', 'charizard', 'squirtle']})
    result = extract_features(df)
    
    feature_names = get_feature_names()
    for feat in feature_names:
        assert feat in result.columns, f"Missing feature: {feat}"


def test_extract_features_name_length():
    """Test name_length calculation."""
    df = pd.DataFrame({'Name': ['aa', 'bbb', 'cccc']})
    result = extract_features(df)
    assert list(result['name_length']) == [2, 3, 4]


def test_extract_features_vowel_ratio():
    """Test vowel ratio calculation."""
    df = pd.DataFrame({'Name': ['aaaa', 'bcdf']})
    result = extract_features(df)
    # aaaa: 4 vowels / 4 chars = 1.0
    # bcdf: 0 vowels / 4 chars = 0.0
    assert pytest.approx(result['vowel_ratio'].iloc[0], rel=0.01) == 1.0
    assert pytest.approx(result['vowel_ratio'].iloc[1], rel=0.01) == 0.0


def test_extract_features_ends_with_vowel():
    """Test ends_with_vowel detection."""
    df = pd.DataFrame({'Name': ['bulbasaur', 'charizard', 'eevee']})
    result = extract_features(df)
    assert result['ends_with_vowel'].tolist() == [0, 0, 1]  # r, d, e


def test_load_and_preprocess_returns_correct_shape():
    """Test data loading returns valid X, y."""
    X, y = load_and_preprocess('data/pokemon.csv')
    assert len(X) > 0
    assert len(y) > 0
    assert X.shape[1] == len(get_feature_names())
    # Type1 should have multiple classes
    assert y.nunique() >= 5


def test_get_feature_names_count():
    """Test that get_feature_names returns exactly 10 features."""
    features = get_feature_names()
    assert len(features) == 10

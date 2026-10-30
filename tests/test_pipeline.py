"""Tests for the training pipeline (probabilities, threshold, scaler, joblib)."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from wine_quality.data import DatasetLoader
from wine_quality.features import NUM_FEATURES
from wine_quality.models import Trainer, load_model

THRESHOLD = 0.5  # decision threshold on P(good)
Bundle = tuple[Pipeline, pd.DataFrame, pd.DataFrame]


@pytest.fixture(scope="module")
def fitted() -> Bundle:
    """Load the data once and fit a logistic regression pipeline on train."""
    loader = DatasetLoader()
    loader.load()
    X_train, X_test, y_train, _ = loader.split()
    pipe = Trainer().make_pipeline(LogisticRegression(max_iter=1000))
    pipe.fit(X_train, y_train)
    return pipe, X_train, X_test


def test_probabilities_between_0_and_1(fitted: Bundle) -> None:
    """predict_proba returns values in [0, 1] that sum to 1 for each wine."""
    pipe, _, X_test = fitted
    proba = pipe.predict_proba(X_test)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert np.allclose(proba.sum(axis=1), 1)


def test_threshold_gives_zeros_and_ones(fitted: Bundle) -> None:
    """Applying the threshold to P(good) gives both 0 and 1 labels."""
    pipe, _, X_test = fitted
    labels = (pipe.predict_proba(X_test)[:, 1] >= THRESHOLD).astype(int)
    assert set(np.unique(labels)) == {0, 1}


def test_scaler_fitted_on_train_only(fitted: Bundle) -> None:
    """The scaler saw only the train rows and learned the train means."""
    pipe, X_train, _ = fitted
    scaler = pipe.named_steps["prep"].named_transformers_["num"]
    assert scaler.n_samples_seen_ == len(X_train)
    assert np.allclose(scaler.mean_, X_train[NUM_FEATURES].mean().to_numpy())


def test_joblib_roundtrip(fitted: Bundle, tmp_path: Path) -> None:
    """A model saved with joblib and reloaded gives the same predictions."""
    pipe, _, X_test = fitted
    path = tmp_path / "model.joblib"
    joblib.dump(pipe, path)
    reloaded = load_model(str(path))
    assert np.array_equal(pipe.predict(X_test), reloaded.predict(X_test))
    assert np.allclose(pipe.predict_proba(X_test), reloaded.predict_proba(X_test))
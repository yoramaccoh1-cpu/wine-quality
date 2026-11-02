"""Tests for the training pipeline (probabilities, threshold, scaler, joblib)."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from wine_quality.data import DatasetLoader
from wine_quality.features import NUM_FEATURES
from wine_quality.models import Trainer, load_model

THRESHOLD = 0.5


def get_fitted_pipe() -> tuple[Pipeline, pd.DataFrame, pd.DataFrame]:
    """Fit a logistic regression pipeline and return it with X_train and X_test."""
    loader = DatasetLoader()
    loader.load()
    X_train, X_test, y_train, _ = loader.split()
    pipe = Trainer().make_pipeline(LogisticRegression(max_iter=1000))
    pipe.fit(X_train, y_train)
    return pipe, X_train, X_test


def test_probabilities_between_0_and_1() -> None:
    """predict_proba returns values in [0, 1]."""
    pipe, _, X_test = get_fitted_pipe()
    proba = pipe.predict_proba(X_test)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_threshold_gives_zeros_and_ones() -> None:
    """Applying the threshold gives only 0s and 1s."""
    pipe, _, X_test = get_fitted_pipe()
    labels = (pipe.predict_proba(X_test)[:, 1] >= THRESHOLD).astype(int)
    assert set(np.unique(labels)) == {0, 1}


def test_scaler_fitted_on_train_only() -> None:
    """The scaler learned only from the train set."""
    pipe, X_train, _ = get_fitted_pipe()
    scaler = pipe.named_steps["prep"].named_transformers_["num"]
    assert scaler.n_samples_seen_ == len(X_train)
    assert np.allclose(scaler.mean_, X_train[NUM_FEATURES].mean().to_numpy())


def test_joblib_roundtrip(tmp_path: Path) -> None:
    """Save and reload gives same predictions."""
    pipe, _, X_test = get_fitted_pipe()
    path = tmp_path / "model.joblib"
    joblib.dump(pipe, path)
    reloaded = load_model(str(path))
    assert np.array_equal(pipe.predict(X_test), reloaded.predict(X_test))
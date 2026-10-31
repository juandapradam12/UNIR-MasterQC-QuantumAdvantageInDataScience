"""Carga y preparacion de datasets para los experimentos."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def make_dataset(
    n_samples: int = 100,
    noise: float = 0.15,
    test_size: float = 0.3,
    seed: int = 42,
    feature_range: tuple[float, float] = (0.0, np.pi),
):
    """Dataset sintetico binario (moons) escalado al rango de codificacion cuantica.

    Devuelve X_train, X_test, y_train, y_test. El escalador se ajusta solo con train.
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = MinMaxScaler(feature_range=feature_range).fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test

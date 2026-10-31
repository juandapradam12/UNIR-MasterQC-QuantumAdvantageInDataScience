"""Experimento 01: SVM clasica (RBF) vs SVM con kernel cuantico en 'moons'."""

from __future__ import annotations

import json
import time
from pathlib import Path

from sklearn.metrics import accuracy_score

from qadvantage.data import make_dataset
from qadvantage.models import classical_svm, quantum_kernel_svm

RESULTS = Path(__file__).resolve().parents[1] / "results"


def run(n_samples: int = 100, seed: int = 42) -> dict:
    X_train, X_test, y_train, y_test = make_dataset(n_samples=n_samples, seed=seed)
    out = {"n_samples": n_samples, "seed": seed}
    for name, model in {
        "svm_rbf": classical_svm(),
        "svm_quantum_kernel": quantum_kernel_svm(n_features=X_train.shape[1]),
    }.items():
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        out[name] = {"test_accuracy": acc, "seconds": time.perf_counter() - t0}
    return out


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "01_baseline_kernel.json").write_text(json.dumps(result, indent=2))

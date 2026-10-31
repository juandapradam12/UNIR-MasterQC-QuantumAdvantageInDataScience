import numpy as np

from qadvantage.data import make_dataset
from qadvantage.models import classical_svm, quantum_kernel_svm


def test_dataset_shapes_and_range():
    X_train, X_test, y_train, y_test = make_dataset(n_samples=40, seed=0)
    assert X_train.shape[1] == X_test.shape[1] == 2
    assert len(X_train) == len(y_train) and len(X_test) == len(y_test)
    assert X_train.min() >= -1e-9 and X_train.max() <= np.pi + 1e-9


def test_quantum_kernel_svm_learns():
    X_train, X_test, y_train, y_test = make_dataset(n_samples=40, seed=0)
    model = quantum_kernel_svm(n_features=2, reps=1).fit(X_train, y_train)
    assert model.score(X_test, y_test) > 0.6


def test_classical_baseline_learns():
    X_train, X_test, y_train, y_test = make_dataset(n_samples=40, seed=0)
    assert classical_svm().fit(X_train, y_train).score(X_test, y_test) > 0.8

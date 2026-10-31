"""Modelos clasicos y cuanticos con una interfaz comun (scikit-learn)."""

from __future__ import annotations

from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from sklearn.svm import SVC


def classical_svm(C: float = 1.0, gamma: str | float = "scale") -> SVC:
    """Baseline clasico: SVM con kernel RBF."""
    return SVC(kernel="rbf", C=C, gamma=gamma)


def quantum_kernel_svm(n_features: int = 2, reps: int = 2, C: float = 1.0) -> SVC:
    """SVM con kernel cuantico (ZZFeatureMap + fidelidad exacta por statevector)."""
    feature_map = zz_feature_map(feature_dimension=n_features, reps=reps)
    kernel = FidelityStatevectorKernel(feature_map=feature_map)
    return SVC(kernel=kernel.evaluate, C=C)

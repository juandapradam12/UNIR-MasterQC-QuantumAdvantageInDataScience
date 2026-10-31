# Plan de trabajo del TFM

## Pregunta de investigacion (borrador)

Bajo que condiciones un modelo cuantico de aprendizaje automatico supera a un baseline clasico
bien ajustado en tareas tipicas de ciencia de datos (clasificacion, regresion, clustering)?

## Decisiones abiertas (a definir con el tutor)

- [ ] Enfoque: kernels cuanticos, clasificadores variacionales, QML generativo, otro.
- [ ] Datasets: sinteticos con estructura cuantica vs. datasets reales tabulares.
- [ ] Hardware: solo simulacion (statevector / ruido con Aer) o tambien QPU real.
- [ ] Framework: Qiskit (actual), PennyLane u otro.
- [ ] Metrica de "ventaja": precision, coste computacional, escalado, complejidad muestral.

## Fases

1. **Estado del arte**: ventaja cuantica en ML (dequantization, Havlicek et al. 2019,
   Huang et al. 2021 "power of data", Schuld & Killoran 2019, barren plateaus).
2. **Baseline y metodologia**: protocolo de evaluacion justo (mismo presupuesto de ajuste,
   validacion cruzada, varias semillas, intervalos de confianza).
3. **Experimentos cuanticos**: kernels (`experiments/01_*`), variacionales, efecto del ruido.
4. **Analisis de escalado**: numero de qubits, profundidad, coste de evaluacion del kernel.
5. **Redaccion de la memoria** en `docs/memoria/`.

## Experimentos

| # | Script | Descripcion | Estado |
|---|--------|-------------|--------|
| 01 | `experiments/01_baseline_kernel.py` | SVM RBF vs SVM con ZZFeatureMap en moons | Esqueleto |

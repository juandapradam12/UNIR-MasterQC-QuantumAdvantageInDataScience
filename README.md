# UNIR-MasterQC-QuantumAdvantageInDataScience

TFM del Master en Computacion Cuantica (UNIR): **ventaja cuantica en ciencia de datos**.

El objetivo es comparar de forma rigurosa modelos cuanticos (p. ej. kernels cuanticos, QML variacional)
frente a baselines clasicos bien ajustados, y analizar cuando, si alguna vez, hay evidencia de ventaja.

## Estructura

```
src/qadvantage/   Codigo reutilizable (datos, modelos, metricas)
experiments/      Scripts de experimentos numerados y reproducibles
notebooks/        Exploracion y figuras
tests/            Tests (pytest)
docs/             Planificacion y borradores de la memoria
data/             raw/ y processed/ (no versionados)
results/          Salidas de los experimentos (figuras versionadas, json/csv no)
```

## Puesta en marcha

```bash
python -m venv .venv && source .venv/bin/activate   # o: uv venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
python experiments/01_baseline_kernel.py
```

## Calidad

```bash
ruff check . && ruff format --check .
pytest
```

## Estado

Ver [`docs/plan.md`](docs/plan.md) para el plan de trabajo y las decisiones abiertas.

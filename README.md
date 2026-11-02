# La Evolución de la Ciencia de Datos hacia la Computación Cuántica: Análisis de la Ventaja Cuántica en Casos Aplicados

Trabajo Fin de Máster · Máster en Computación Cuántica · UNIR<br>
Autor: Juan David Prada Malagón · Director: Ricardo Rafael Domínguez Reyes

> **¿Mejoran los métodos cuánticos, simulados con pocos qubits, la calidad o el coste frente a los clásicos en problemas de ciencia de datos?**
>
> En este régimen, no. En seis casos de uso con un mismo protocolo, ningún modelo o solver cuántico supera al mejor clásico. Cuando la calidad empata, el método cuántico tarda entre dos y cuatro órdenes de magnitud más.

| | |
|---|---|
| Memoria | [`Documento/main.pdf`](Documento/main.pdf) |
| Defensa | [`Defensa/defensa.pdf`](Defensa/defensa.pdf) |
| Experimentos | [`Notebooks/`](Notebooks/) |

---

## 1. Por qué este trabajo

La teoría promete kernels cuánticos en espacios de Hilbert de dimensión exponencial (Havlíček et al., 2019) y algoritmos variacionales como QAOA y VQE pensados para hardware NISQ (Farhi et al., 2014; Peruzzo et al., 2014). La evidencia empírica es más modesta: con un protocolo justo, los modelos clásicos de referencia suelen igualar o superar a los cuánticos en datasets pequeños (Bowles et al., 2024), y la ventaja de un kernel cuántico depende de cómo encaja su geometría con la del dato (Huang et al., 2021).

El problema habitual de las comparaciones es la asimetría: preprocesos distintos para cada rama, baselines poco ajustados o tiempos que no se reportan. Este trabajo compara las dos familias **con el mismo dato, el mismo preproceso, las mismas métricas y el tiempo medido**, y separa lo que el experimento demuestra de lo que no permite afirmar.

## 2. Seis casos, un mismo protocolo

| Familia | Caso | Tamaño | Qué representa |
|---|---|---|---|
| Clasificación | Iris | 150 muestras, 4 variables, 3 clases | dato convencional, una clase separada y dos solapadas |
| Clasificación | Bars and Stripes | 1200 patrones $4\times4$, 2 clases | dato convencional binario, comprimido a 4 componentes |
| Clasificación | Ising 1D | 200 estados fundamentales, 2 fases | dato de sistema cuántico |
| Optimización | TSP | 5 nodos, 6 rutas candidatas | problema con restricción de selección única |
| Optimización | Minimum Dominating Set | 6 vértices | problema con restricción de cobertura |
| Optimización | MaxCut (HamLib) | 6 vértices, 12 aristas | caso canónico de QAOA, sin restricciones |

<table>
<tr>
<td align="center"><img src="Notebooks/Classification/IrisClassification/figures/svm_iris_pca_scatter.png" width="260"><br><sub>Iris</sub></td>
<td align="center"><img src="Notebooks/Classification/BSClassification/figures/svm_bs_samples.png" width="260"><br><sub>Bars and Stripes</sub></td>
<td align="center"><img src="Notebooks/Classification/IsingClassification/figures/svm_ising_chain.png" width="260"><br><sub>Cadena de Ising de 4 espines</sub></td>
</tr>
<tr>
<td align="center"><img src="Notebooks/Optimization/TSPOptimization/figures/instance_tsp_graph.png" width="260"><br><sub>TSP</sub></td>
<td align="center"><img src="Notebooks/Optimization/MinimumDominatingSetOptimization/figures/instance_mds_graph.png" width="260"><br><sub>MDS</sub></td>
<td align="center"><img src="Notebooks/Optimization/MaxCutOptimization/figures/instance_maxcut_graph.png" width="260"><br><sub>MaxCut (HamLib)</sub></td>
</tr>
</table>

Los modelos y solvers comparados son:

| Familia | Clásico | Cuántico o cuántico-inspirado |
|---|---|---|
| Clasificación | SVM (RBF), Random Forest, XGBoost | QSVM (mapa ZZ), QNN híbrida, VQC |
| Optimización | fuerza bruta, greedy, ILP (PuLP + CBC) | QUBO + Simulated Annealing, QAOA, VQE |

Todo el cómputo cuántico es **simulación de estado sobre CPU**: 4 qubits en clasificación y 6 en optimización.

## 3. La representación como eje

Un método cuántico no opera sobre el dato, sino sobre una representación de él. En clasificación, esa representación es el **feature map**; en optimización, la reformulación **QUBO/Ising**. El trabajo trata esa transformación como un eje explícito de análisis.

### Clasificación: del dato al espacio de Hilbert

Tras la partición estratificada (60/40, semilla 123), cuando $d>4$ se aplica PCA a 4 componentes, y después un escalado $\mathrm{MinMax}[0,\pi]$. El mismo $\mathbf{x}$ alimenta a los seis modelos. En la rama cuántica, un circuito lleva cada muestra a un estado de 4 qubits:

```math
\phi:\ \mathbb{R}^4 \longrightarrow \mathcal{H}=\left(\mathbb{C}^2\right)^{\otimes 4}\simeq\mathbb{C}^{16},
\qquad
|\phi(\mathbf{x})\rangle = U_\phi(\mathbf{x})\,|0\rangle^{\otimes 4}.
```

El QSVM usa el mapa ZZ de Qiskit (dos repeticiones, entrelazamiento completo), que alterna capas de Hadamard con fases dependientes del dato:

```math
U_\phi(\mathbf{x}) = \Big[\,U_Z(\mathbf{x})\,H^{\otimes 4}\Big]^2,
\qquad
U_Z(\mathbf{x}) = \exp\!\Big(-i\sum_{i} x_i Z_i \;-\; i\sum_{i \lt j}(\pi-x_i)(\pi-x_j)\,Z_iZ_j\Big),
```

y clasifica con un SVM sobre el kernel de fidelidad:

```math
K(\mathbf{x},\mathbf{x}') = \big|\langle \phi(\mathbf{x}) \,|\, \phi(\mathbf{x}')\rangle\big|^2 .
```

La QNN y el VQC usan codificación angular, $U_\phi(\mathbf{x})=\bigotimes_i R_\sigma(x_i)$, seguida de un ansatz entrenable, y clasifican a partir de valores esperados $\langle Z_i\rangle$.

### El dato de sistema: Ising

Las muestras de Ising son estados fundamentales del modelo de campo transversal en cadenas abierta y cerrada de 4 espines, con $J=-1$ y 100 valores de $h\in[0,3]$ por cadena:

```math
H(h) = J\sum_{i} Z_i Z_{i+1} \;+\; h\sum_i X_i,
\qquad
y = \mathbf{1}\big[\,h > |J|\,\big].
```

Las features son las $2^4$ amplitudes complejas del estado (32 valores reales), reducidas por PCA a 4 componentes que retienen el 99.9 % de la varianza. El campo $h$ no entra como feature, para evitar la fuga de la etiqueta.

### Optimización: del problema al Hamiltoniano

Los tres problemas se escriben como minimización de una forma cuadrática binaria. Con $`x_i\in\{0,1\}`$ y $`z_i = 1-2x_i \in \{\pm1\}`$, cada QUBO equivale a la energía de un Hamiltoniano de espines (Lucas, 2014):

```math
\min_{\mathbf{x}\in\{0,1\}^n} \mathbf{x}^\top Q\,\mathbf{x}
\quad\Longleftrightarrow\quad
\min_{\mathbf{z}\in\{\pm1\}^n} H_C(\mathbf{z}) = \sum_{i \lt j} J_{ij}\, z_i z_j + \sum_i h_i\, z_i + \text{cte}.
```

| Problema | QUBO / Ising usado |
|---|---|
| TSP | $`\displaystyle \sum_{r} c_r\, y_r + P\Big(1-\sum_r y_r\Big)^2`$, una variable por ruta candidata, $P=100$ |
| MDS | $\displaystyle A\sum_v x_v + B\sum_v\Big(1-\sum_{u\in N[v]}x_u\Big)^2$, con $A=1$, $B=2$ |
| MaxCut | $\displaystyle C(\mathbf{z})=\sum_{(i,j)\in E}\frac{1-z_iz_j}{2}$, sin penalizaciones |

QAOA prepara el estado variacional alternando el Hamiltoniano de coste y un mezclador $B=\sum_i X_i$, con $p$ capas:

```math
|\boldsymbol\gamma,\boldsymbol\beta\rangle = \prod_{k=1}^{p} e^{-i\beta_k B}\, e^{-i\gamma_k H_C}\; |+\rangle^{\otimes n},
\qquad
(\boldsymbol\gamma^\star,\boldsymbol\beta^\star) = \arg\min_{\boldsymbol\gamma,\boldsymbol\beta}\ \langle \boldsymbol\gamma,\boldsymbol\beta|\,H_C\,|\boldsymbol\gamma,\boldsymbol\beta\rangle .
```

SA, QAOA y VQE comparten exactamente la misma energía; la diferencia está en cómo buscan su mínimo.

## 4. Clasificación: el cuántico solo empata en el caso más fácil

<p align="center">
<img src="Notebooks/Classification/ResultsClassification/figures/cls_classical_vs_quantum_by_dataset.png" width="640">
</p>

La accuracy media de la familia cuántica queda **0.083 por debajo de la clásica en Iris y 0.101 en Bars and Stripes**; en Ising ambas medias coinciden (0.971). Ising es también el dataset en el que los seis modelos alcanzan al menos 0.95 de accuracy: la paridad aparece donde el problema ya es fácil para todos.

| F1 de test | Iris | Bars and Stripes | Ising | Media | Entrenamiento medio |
|---|---:|---:|---:|---:|---:|
| **SVM (RBF)** | **0.967** | **0.870** | **0.985** | **0.940** | **2 ms** |
| Random Forest | 0.950 | 0.854 | 0.969 | 0.925 | 63 ms |
| XGBoost | 0.950 | 0.849 | 0.937 | 0.912 | 94 ms |
| QSVM | 0.730 | 0.758 | **0.985** | 0.825 | 0.53 s |
| QNN | 0.933 | 0.815 | 0.937 | 0.895 | 2.6 s |
| VQC | 0.950 | 0.688 | 0.969 | 0.869 | 24.4 s |

**El kernel ZZ sobreajusta en los datasets convencionales.** En Iris, el QSVM acierta el 96.7 % del train y solo el 73.3 % del test: el mayor gap de generalización del corpus (0.233). Comete 16 errores de 60, también en setosa, la clase que los otros cinco modelos clasifican sin error. Sobre el plano de las dos primeras componentes, su frontera se fragmenta en islas donde la del SVM traza tres regiones suaves:

<table>
<tr>
<td align="center"><img src="Notebooks/Classification/IrisClassification/figures/svm_iris_pca_boundary.png" width="280"><br><sub>SVM-RBF: F1 0.967, 2 errores en el solape versicolor/virginica</sub></td>
<td align="center"><img src="Notebooks/Classification/IrisClassification/figures/rf_iris_pca_boundary.png" width="280"><br><sub>Random Forest: F1 0.950</sub></td>
<td align="center"><img src="Notebooks/Classification/IrisClassification/figures/xgb_iris_pca_boundary.png" width="280"><br><sub>XGBoost: F1 0.950</sub></td>
</tr>
<tr>
<td align="center"><img src="Notebooks/Classification/IrisClassification/figures/qsvm_iris_pca_boundary.png" width="280"><br><sub>QSVM-ZZ: F1 0.730, 16 errores</sub></td>
<td align="center"><img src="Notebooks/Classification/IrisClassification/figures/qnn_iris_pca_boundary.png" width="280"><br><sub>QNN: F1 0.933</sub></td>
<td align="center"><img src="Notebooks/Classification/IrisClassification/figures/vqc_iris_pca_boundary.png" width="280"><br><sub>VQC: F1 0.950</sub></td>
</tr>
</table>

<sub>Fronteras de decisión en Iris. Fila superior: modelos clásicos; fila inferior: cuánticos. Cada panel evalúa el modelo entrenado sobre el plano PC1–PC2, con las otras dos componentes fijas en la media del train; los círculos marcan los errores reales de test.</sub>

Los árboles trazan cortes casi rectos y la QNN y el VQC, fronteras suaves; solo el QSVM fragmenta el plano. En Bars and Stripes, las dos primeras componentes no separan las clases: el SVM asigna casi todo el plano a una sola clase y aun así acierta el 87 % del test, porque su decisión depende de la tercera y la cuarta componentes. Sobre ese mismo plano, el QSVM vuelve a dividir el espacio en islas:

<table>
<tr>
<td align="center"><img src="Notebooks/Classification/BSClassification/figures/svm_bs_pca_boundary.png" width="280"><br><sub>SVM-RBF: F1 0.870</sub></td>
<td align="center"><img src="Notebooks/Classification/BSClassification/figures/rf_bs_pca_boundary.png" width="280"><br><sub>Random Forest: F1 0.854</sub></td>
<td align="center"><img src="Notebooks/Classification/BSClassification/figures/xgb_bs_pca_boundary.png" width="280"><br><sub>XGBoost: F1 0.849</sub></td>
</tr>
<tr>
<td align="center"><img src="Notebooks/Classification/BSClassification/figures/qsvm_bs_pca_boundary.png" width="280"><br><sub>QSVM-ZZ: F1 0.758</sub></td>
<td align="center"><img src="Notebooks/Classification/BSClassification/figures/qnn_bs_pca_boundary.png" width="280"><br><sub>QNN: F1 0.815</sub></td>
<td align="center"><img src="Notebooks/Classification/BSClassification/figures/vqc_bs_pca_boundary.png" width="280"><br><sub>VQC: F1 0.688</sub></td>
</tr>
</table>

<sub>Fronteras de decisión en Bars and Stripes, con la misma construcción. El plano PC1–PC2 solo muestra una sección de cada frontera.</sub>

**El empate en Ising no demuestra afinidad física.** Sería tentador leerlo como que el mapa ZZ captura la estructura $Z_iZ_{i+1}$ del Hamiltoniano. El experimento no lo permite: el mapa actúa sobre cuatro componentes principales de las amplitudes, no sobre los espines, y todos los modelos resuelven el dataset.

**El coste cambia de escala.** Ningún modelo cuántico está en la frontera calidad–coste: el SVM tiene a la vez el mejor F1 medio y el menor tiempo de entrenamiento. La QNN, el mejor cuántico en media, entrena tres órdenes de magnitud más despacio, y el VQC cuatro. En predicción, QSVM y VQC son de dos a tres órdenes más lentos que los clásicos.

<p align="center">
<img src="Notebooks/Classification/ResultsClassification/figures/cls_performance_vs_cost.png" width="560">
</p>

## 5. Optimización: casi todos llegan al óptimo, a distinto coste

<p align="center">
<img src="Notebooks/Optimization/ResultsOptimization/figures/opt_success_vs_runtime_mean.png" width="560">
</p>

| Success rate / tiempo medio | TSP ($C^\star=16$) | MDS ($C^\star=2$) | MaxCut ($C^\star=8$) |
|---|---:|---:|---:|
| Fuerza bruta | 1.00 · 60 µs | 1.00 · 0.2 ms | 1.00 · 0.3 ms |
| Greedy | 1.00 · 90 µs | **0.00** · 40 µs | 1.00 · 60 µs |
| ILP | 1.00 · 0.07 s | 1.00 · 0.05 s | 1.00 · 0.66 s |
| QUBO + SA | **0.90** · 17 ms | 1.00 · 7 ms | 1.00 · 3 ms |
| QAOA | 1.00 · 1.1 s | **0.875** · 12.6 s | 1.00 · 0.64 s |
| VQE | 1.00 · 2.0 s | 1.00 · 1.6 s | 1.00 · 0.98 s |

Con $n\le 6$, la enumeración recorre los $2^6=64$ bitstrings en menos de un milisegundo y es la referencia, no un baseline débil. Solo fallan tres casos, y cada uno por un motivo distinto:

| Fallo | Qué ocurre |
|---|---|
| Greedy en MDS | Elige primero el vértice de mayor cobertura y no deshace esa elección: tamaño 3 frente al óptimo 2 |
| SA en TSP (2 de 20 semillas) | Con una variable por ruta, todo volteo desde una ruta factible suma la penalización de 100: ambas rutas factibles son mínimos locales |
| QAOA en MDS (1 de 8 semillas) | El optimizador termina en parámetros de mayor energía y devuelve un conjunto que deja un vértice sin dominar |

<table>
<tr>
<td align="center"><img src="Notebooks/Optimization/MinimumDominatingSetOptimization/figures/greedy_mds_dominating_set.png" width="400"><br><sub>Greedy: tamaño 3, no óptimo</sub></td>
<td align="center"><img src="Notebooks/Optimization/MinimumDominatingSetOptimization/figures/qaoa_mds_dominating_set.png" width="400"><br><sub>QAOA: óptimo {Ana, Enrique} en 7 de 8 semillas, 12.6 s por ejecución</sub></td>
</tr>
</table>

El fallo de QAOA no es de la formulación: el SA resuelve **el mismo QUBO** en las 20 semillas en 7 ms. En MDS, la probabilidad de medir el óptimo sigue a la energía alcanzada por el optimizador: es nula en la semilla fallida y está entre 0.62 y 0.75 en las seis mejores (media 0.53).

### Un éxito de 1 no siempre prueba que el circuito funcione

En TSP y MDS, QAOA y VQE devuelven el bitstring de menor energía entre miles de muestras. Con $N$ muestras uniformes sobre 64 bitstrings y un único óptimo, la probabilidad de no verlo nunca es

```math
\Pr[\text{óptimo ausente}] = \left(1-\tfrac{1}{64}\right)^{N},
\qquad
N=3000 \;\Rightarrow\; \approx e^{-47}\approx 0 .
```

Un muestreador uniforme también "acertaría". Por eso, la medida que distingue a un solver variacional es la probabilidad que concentra en el óptimo, no su success rate. Donde se puede medir, sí hay concentración: en MaxCut, con los parámetros finales guardados, el estado QAOA asigna **probabilidad 0.93 a los cortes óptimos, frente a 0.28 de una muestra uniforme**.

<p align="center">
<img src="Notebooks/Optimization/MaxCutOptimization/figures/qaoa_maxcut_cut.png" width="380">
</p>

## 6. Qué se puede afirmar y qué no

| Se puede afirmar | No se puede afirmar |
|---|---|
| Con 4–6 qubits simulados, ningún modelo o solver cuántico supera al mejor clásico | Que no exista ventaja cuántica a mayor escala |
| La paridad de calidad, cuando aparece, llega con un coste temporal mayor | Que el empate en Ising se deba a la física de la cadena |
| El mapa ZZ generaliza peor que un RBF en Iris y Bars and Stripes | Cuánto del coste se debe a la simulación y cuánto al método |
| QAOA concentra probabilidad en MaxCut (0.93 frente a 0.28) | Que QAOA y VQE concentren probabilidad en TSP y MDS a partir de su success rate |

El régimen mide **cercanía al clásico, no ventaja asintótica** (Preskill, 2018; Thanasilp et al., 2024). El coste de simular un circuito crece como $2^n$, y las preguntas que quedan abiertas exigen escala: cadenas de Ising de 8 y 16 espines codificadas directamente en qubits, instancias QUBO mayores con $p$ creciente, y ejecución en hardware real.

## 7. Estructura del repositorio

```text
.
├── Notebooks/
│   ├── lib/                      utilidades comunes (rutas y figuras)
│   ├── Classification/
│   │   ├── IrisClassification/   un notebook por modelo + metrics/ + figures/
│   │   ├── BSClassification/
│   │   ├── IsingClassification/
│   │   └── ResultsClassification/   agrega métricas y genera las comparativas
│   └── Optimization/
│       ├── TSPOptimization/      un notebook por solver + metrics/ + figures/
│       ├── MinimumDominatingSetOptimization/
│       ├── MaxCutOptimization/   incluye instance/hamlib_maxcut.json
│       └── ResultsOptimization/
├── Documento/                    memoria en LaTeX (main.tex, Capitulos/, Estilos/, Imagenes/)
├── Defensa/                      diapositivas de la defensa en Beamer
└── requirements.txt              dependencias con versión fijada
```

Cada notebook de modelo o solver escribe sus métricas en JSON. Los notebooks de `Results*` solo leen esos JSON y generan las tablas y figuras comparativas; todas las cifras de la memoria, de la defensa y de este README salen de ahí.

## 8. Reproducción

```bash
python3.14 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install qiskit-machine-learning        # no está fijado en requirements.txt
```

1. Ejecuta los notebooks de cada carpeta de caso, en cualquier orden. Cada uno escribe su JSON en `metrics/` y sus figuras en `figures/`.
2. Ejecuta los `Results_*.ipynb` de `ResultsClassification/` y `ResultsOptimization/`.
3. Compila la memoria y la defensa con dos pasadas de `pdflatex`:

```bash
cd Documento && pdflatex main.tex && pdflatex main.tex
cd ../Defensa && pdflatex defensa.tex && pdflatex defensa.tex
```

Las semillas están fijadas en las particiones y en los solvers estocásticos. Los tiempos son wall-clock de una sola máquina y solo son comparables en orden de magnitud.

## Referencias principales

- Bowles, J., Ahmed, S., & Schuld, M. (2024). *Better than classical? The subtle art of benchmarking quantum machine learning models*. [arXiv:2403.07059](https://doi.org/10.48550/arXiv.2403.07059)
- Farhi, E., Goldstone, J., & Gutmann, S. (2014). *A quantum approximate optimization algorithm*. [arXiv:1411.4028](https://doi.org/10.48550/arXiv.1411.4028)
- Havlíček, V., et al. (2019). *Supervised learning with quantum-enhanced feature spaces*. Nature, 567, 209–212. [doi:10.1038/s41586-019-0980-2](https://doi.org/10.1038/s41586-019-0980-2)
- Huang, H.-Y., et al. (2021). *Power of data in quantum machine learning*. Nature Communications, 12, 2631. [doi:10.1038/s41467-021-22539-9](https://doi.org/10.1038/s41467-021-22539-9)
- Lucas, A. (2014). *Ising formulations of many NP problems*. Frontiers in Physics, 2, 5. [doi:10.3389/fphy.2014.00005](https://doi.org/10.3389/fphy.2014.00005)
- Peruzzo, A., et al. (2014). *A variational eigenvalue solver on a photonic quantum processor*. Nature Communications, 5, 4213. [doi:10.1038/ncomms5213](https://doi.org/10.1038/ncomms5213)
- Preskill, J. (2018). *Quantum computing in the NISQ era and beyond*. Quantum, 2, 79. [doi:10.22331/q-2018-08-06-79](https://doi.org/10.22331/q-2018-08-06-79)
- Sawaya, N. P. D., et al. (2024). *HamLib: A library of Hamiltonians for benchmarking quantum algorithms and hardware*. Quantum, 8, 1559. [doi:10.22331/q-2024-12-11-1559](https://doi.org/10.22331/q-2024-12-11-1559)
- Thanasilp, S., Wang, S., Cerezo, M., & Holmes, Z. (2024). *Exponential concentration in quantum kernel methods*. Nature Communications, 15, 5200. [doi:10.1038/s41467-024-49287-w](https://doi.org/10.1038/s41467-024-49287-w)

La bibliografía completa está en la memoria.

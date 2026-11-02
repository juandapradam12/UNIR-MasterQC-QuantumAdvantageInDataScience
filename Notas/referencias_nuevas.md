# Referencias nuevas (lectura) — enriquecer Documento/

Conservar todas las de `Documento/Capitulos/referencias.tex`. Esta lista es material adicional; prioridad = utilidad para Iris/BS/Ising + TSP/MDS/MaxCut y el eje representación.

## Alta prioridad (citar en contexto / desarrollo / evaluación)

| Fuente | Qué es | Dónde encaja |
|---|---|---|
| [Bowles, Ahmed, Schuld 2024](https://arxiv.org/abs/2403.07059) *Better than classical?* + [qml-benchmarks](https://github.com/XanaduAI/qml-benchmarks) | Benchmark grande: clásicos suelen ganar; quitar entrelazamiento a veces no empeora; BS entre tareas | Contexto QML, justificación del protocolo justo, lectura “sin ventaja en demos pequeñas” |
| [HamLib](https://arxiv.org/abs/2306.13126) Sawaya et al. | Librería de Hamiltonianos: MaxCut, TSP, … | Contexto/métodos MaxCut (y TSP HamLib si se cita) |
| PennyLane [Bars and Stripes](https://pennylane.ai/datasets/bars-and-stripes) | Dataset convencional para QML | Datasets clasificación |
| PennyLane [Ising / qspin](https://pennylane.ai/datasets/ising) | Datos de sistema (fases) | Contraste classical vs quantum data |
| PennyLane [what is a quantum dataset](https://pennylane.ai/datasets/what-is-a-quantum-dataset) / [conventional datasets](https://pennylane.ai/datasets/conventional-datasets-for-qml) | Taxonomía dato cuántico vs convencional | Contexto §datasets (ya en Notas §2–3) |
| PennyLane [hamlib-maxcut](https://pennylane.ai/datasets/hamlib-maxcut) | Instancia usada en el TFM | Métodos MaxCut |
| PennyLane [QAOA MaxCut demo](https://pennylane.ai/demos/tutorial_qaoa_maxcut) | Circuito QAOA estilo Farhi | Desarrollo MaxCut / QAOA |
| PennyLane [QAOA intro](https://pennylane.ai/demos/tutorial_qaoa_intro) | Intro QAOA | Contexto optimización |
| PennyLane [variational classifier](https://pennylane.ai/demos/tutorial_variational_classifier) | VQC + nodos clásicos/cuánticos | Desarrollo QNN/VQC |
| PennyLane [embedding blog](https://pennylane.ai/blog/2022/08/how-to-embed-data-into-a-quantum-state/) + [glossary embedding](https://pennylane.ai/glossary/quantum_embedding) | Basis / angle / amplitude (+ QAOA/IQP templates) | Eje representación |
| sklearn [Iris](https://scikit-learn.org/1.5/auto_examples/datasets/plot_iris_dataset.html) | Iris canónico | Datasets |
| [ACM TQC 10.1145/3678184](https://dl.acm.org/doi/10.1145/3678184) Lubinski et al. | Benchmarks de optimización (MaxCut, calidad vs tiempo, QAOA/annealing) | Evaluación opt: métricas effort/quality |
| [Somma et al. arXiv:0712.1008](https://arxiv.org/pdf/0712.1008) *Quantum Simulated Annealing* | Speedup teórico SA cuántico vs clásico en gap | Contexto SA / no confundir con SA clásico del TFM |
| Farhi et al. 2014 (ya en bib) vía demos QAOA | QAOA canónico | Ya citado |

## Media prioridad (apoyo / futuro / matices)

| Fuente | Qué es | Dónde encaja |
|---|---|---|
| PennyLane [hamlib-TSP](https://pennylane.ai/datasets/hamlib-travelling-salesperson-problem) | TSP en HamLib | Trabajo futuro / contraste con TSP propio |
| PennyLane [transfer learning](https://pennylane.ai/demos/tutorial_quantum_transfer_learning) | Transfer learning cuántico | Futuro / extensiones |
| PennyLane [QFT and groups](https://pennylane.ai/demos/tutorial_qft_and_groups) | QFT | Solo si se habla de transformadas; no central |
| PennyLane [Grover demo](https://pennylane.ai/demos/tutorial_grovers_algorithm) | Grover pedagógico | Contexto histórico (ya hay Grover en bib) |
| [PRX 9, 041029](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.9.041029) Sentís et al. | Clustering no supervisado de *datos cuánticos* (estados) | Extensiones clustering; no es Acc/F1 Iris |
| [arXiv:2502.15129](https://arxiv.org/abs/2502.15129) de Paula Neto | Complejidad del dato → recomendar arquitectura de circuito | Representación / elección de ansatz |
| [PRResearch rph8-g15q](https://journals.aps.org/prresearch/abstract/10.1103/rph8-g15q) Parzygnat et al. | *Toward structure-preserving quantum encodings* | Eje representación (encoding que preserve estructura) |
| [arXiv:2609.16171](https://arxiv.org/html/2609.16171v1) Rovara et al. | Compilación / programas cuánticos estructurados (no QML de datos) | Poco uso en este TFM; opcional hardware/software futuro |
| [arXiv:2607.01197](https://arxiv.org/html/2607.01197v2) Yu et al. + [repo QML](https://github.com/Z-537-437/QML) | Comparación empírica Q vs C (clasificación + RL): clásico gana en Acc/tiempo | Refuerza la lectura del TFM; citar con cautela (preprint reciente) |
| CERN/Atos Fast-Start PDF | Material docente / intro | Contexto pedagogía; no evidencia empírica |
| Gurobi FAQ ILP | Definición ILP | Métodos solvers clásicos (una frase) |
| [ACM 10.1145/1273496.1273497](https://dl.acm.org/doi/pdf/10.1145/1273496.1273497) | (ML clásico histórico — verificar título al citar) | Solo si se usa en baselines |
| GitHub [qml-benchmarks](https://github.com/XanaduAI/qml-benchmarks) | Código Bowles et al. | Reproducibilidad |
| GitHub [Z-537-437/QML](https://github.com/Z-537-437/QML), [Quantum-Linear-Regression](https://github.com/amey-kulkarni27/Quantum-Linear-Regression) | Demos/repos | Futuro regresión; no corpus del TFM |

## Baja prioridad / no citar como evidencia

| Fuente | Motivo |
|---|---|
| Medium “classical vs quantum ML…”, “complete guide to quantum data encoding” | Blog no peer-reviewed; puede orientar vocabulario, no claims |
| vocal.media “quantum algorithms outperform…” | Clickbait; choca con Bowles y con nuestros resultados |
| quantumzeitgeist (QML hybrid, beginners, FBI…) | Divulgación / noticias; no base de tesis |
| Duplicados BS / conventional / HamLib-TSP / 2306.13126 | Ya cubiertos arriba |

## Ideas concretas para el documento (cuando toque cada sección)

1. **Contexto:** Bowles → la literatura sobreestima “outperform”; hace falta protocolo justo. HamLib → por qué MaxCut/TSP estandarizados. PennyLane quantum vs conventional dataset → Iris/BS clásicos; Ising sistema.
2. **Desarrollo:** demos VQC / QAOA MaxCut / embeddings (angle, ZZ) como ancla técnica; sklearn Iris; HamLib maxcut dataset.
3. **Evaluación:** ACM Lubinski (calidad vs tiempo en MaxCut) + nuestros CSV; alinear narrativa con Bowles (clásico gana en escala pequeña).
4. **Representación:** blog/glossary embedding + Bowles (entanglement no siempre ayuda) + opcional de Paula Neto.
5. **Futuro:** HamLib-TSP, transfer learning, repos de regresión lineal cuántica; no hinchar el corpus actual.

## Ya en bibliografía (no duplicar)

Preskill, Farhi, Biamonte, Havlíček, Huang, Schuld, Cerezo, Thanasilp, Lucas, Grover, Peruzzo, McClean, etc. en `referencias.tex`.

## Claves `\cite{...}` añadidas en `referencias.tex`

| Clave | Fuente |
|---|---|
| `aimeur2007` | Quantum clustering (ICML) |
| `bowles2024` | Better than classical? |
| `cernAtos2021` | CERN–Atos Fast Start PDF |
| `dePaulaNeto2025` | Data complexity → circuit recommendation |
| `gurobiIlp` | FAQ ILP |
| `kulkarniQlr` | Quantum linear regression (GitHub) |
| `lubinski2024` | ACM opt. benchmarks / MaxCut |
| `parateEncodingGuide` | Medium encoding guide |
| `parzygnat2025` | Structure-preserving encodings |
| `pennylaneBarsAndStripes` | Dataset BS |
| `pennylaneConventionalDatasets` | Conventional datasets for QML |
| `pennylaneEmbedBlog` | How to embed data… |
| `pennylaneGlossaryEmbedding` | Glossary embedding |
| `pennylaneGrover` | Demo Grover |
| `pennylaneHamlibMaxcut` | Dataset HamLib MaxCut |
| `pennylaneHamlibTsp` | Dataset HamLib TSP |
| `pennylaneIsing` | Dataset Ising |
| `pennylaneQaoaIntro` | Demo QAOA intro |
| `pennylaneQaoaMaxcut` | Demo QAOA MaxCut |
| `pennylaneQftGroups` | Demo QFT |
| `pennylaneTransferLearning` | Demo transfer learning |
| `pennylaneVariationalClassifier` | Demo VQC |
| `pennylaneWhatIsQuantumDataset` | What is a quantum dataset? |
| `qzHybridQml` / `qzBeginners` / `qzFbi` | Quantum Zeitgeist |
| `rovara2026` | Structured quantum programs |
| `sawaya2024hamlib` | HamLib paper |
| `sklearnIris` | Iris example sklearn |
| `sentis2019` | Unsupervised classification of quantum data |
| `somma2008` | Quantum simulated annealing |
| `vocalQuantumOutperform` | Vocal Media |
| `wangClassicalVsQuantumMl` | Medium C vs Q ML |
| `qmlBenchmarks` | Xanadu qml-benchmarks repo |
| `yu2026` | Unified empirical Q vs C |
| `githubZ537Qml` | Repo Yu et al. |

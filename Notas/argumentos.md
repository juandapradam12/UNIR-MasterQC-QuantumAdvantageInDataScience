# Argumentos para el documento y la defensa

Cada argumento va con: afirmación, desarrollo, qué falta por verificar y dónde se usará.
Todo lo marcado como **por verificar** no debe citarse hasta comprobar la fuente.

---

## 1. Con pocos qubits, un experimento de kernel cuántico no puede mostrar ventaja

**Afirmación.** En el régimen de pocos qubits (p. ej. 4) un experimento de QSVM no puede demostrar ventaja cuántica. Solo puede mostrar si el kernel cuántico se acerca al clásico. La ventaja solo es plausible con muchos qubits, y ahí ya no se puede simular ni ejecutar de forma fiable.

**Desarrollo.**

1. *El feature map usa un qubit por característica.* Con el feature map ZZ (`get_quantum_state` en `Notebooks/Classification/QSVM vs SVM.ipynb`), cada característica ocupa un qubit. Si los datos tienen 16 características y se simulan 4 qubits, hay que comprimir con PCA. Esa pérdida de información afecta por igual al modelo cuántico y al clásico, que recibe las mismas 4 características.
2. *Por qué los qubits limitan:*
   - **Simulación.** El estado tiene \(2^n\) amplitudes. Con 16 qubits son ~65.000 por punto, y con 30 qubits un solo estado ocupa 16 GiB (\(2^{30} \times 16\) bytes, amplitudes complejas de 128 bits).
   - **Hardware real.** Hay pocos qubits útiles y mucho ruido. Con entrelazamiento completo, el número de puertas de dos qubits crece como \(n^2\), y el ruido acumulado destruye la señal en circuitos profundos.
   - **Más qubits no es mejor sin más.** Con muchos qubits el kernel puede concentrarse (todos los puntos parecen igual de distantes) y el modelo deja de generalizar (*exponential concentration*).
3. *La paradoja.* Con pocos qubits, un ordenador clásico simula el kernel sin esfuerzo, así que no hay ventaja posible. Con muchos qubits, la ventaja es plausible, pero ya no se puede simular ni ejecutar con fiabilidad.
4. *Consecuencia metodológica.* Hay que explicar por qué se trabaja en el régimen pequeño y qué se puede concluir y qué no. Una comparación en 4 qubits es una prueba de que el pipeline funciona y de cuánto se acerca el kernel cuántico al clásico. No es una prueba de ventaja.

**Argumentos relacionados (mismo capítulo).**

- *El feature map ZZ no es arbitrario, pero tiene elecciones sin justificar para cada dataset.* Las puertas Hadamard, `RZ` y ZZ tienen cada una una función (superposición, codificación del dato, entrelazamiento). Sin la puerta ZZ el kernel se factoriza en un producto de kernels de un qubit y es trivial de calcular clásicamente. Pero el ángulo \((\pi - x_i)(\pi - x_j)\), el número de repeticiones y el escalado a \([0, \pi]\) son elecciones heredadas del artículo original, no derivadas de los datos.
- *Que un circuito sea difícil de simular no implica que sirva para aprender.* La dureza de simulación es una propiedad del circuito. La utilidad para clasificar depende de la relación entre el kernel y la estructura de los datos.
- *El kernel cuántico del notebook no tiene parámetro de ancho de banda.* El RBF clásico se adapta a la escala de los datos con `gamma`. El kernel cuántico está fijado por el escalado y las repeticiones. Una comparación justa ajusta ambos, y en el cuántico el análogo de `gamma` es un factor de escala de las entradas.

**Por verificar.**

- Cita para *exponential concentration* (Thanasilp et al., 2022 es mi candidato; confirmar).
- Cita para "dureza de simulación no implica utilidad para aprender": Huang et al. (2021), *Power of data in quantum machine learning*. Confirmar que dice esto.
- Origen del dataset *Bars and Stripes* de PennyLane: creo que viene de Bowles, Ahmed y Schuld (2024), *Better than classical?*, pero no está confirmado.

**Dónde se usa.** Documento: capítulo de contexto (régimen NISQ) y capítulo de evaluación (limitaciones). Defensa: diapositiva de limitaciones y respuesta a "¿por qué solo 4 qubits?".

---

## 2. Distinción entre datos cuánticos y datos clásicos (convencionales) para QML

**Afirmación.** En la literatura de PennyLane, un *quantum dataset* no es “cualquier dataset usado en un circuito”, sino una colección de **datos que describen sistemas cuánticos y su evolución** (entradas/salidas de una función cuántica). En paralelo, el QML también trabaja con **datos convencionales / clásicos** (imágenes, tablas, series, etiquetas), que hay que **embeber** en un estado cuántico. Confundir ambos sesga la interpretación de “ventaja” y de qué se está midiendo en el TFM.

**Fuentes (base de esta nota).**

- PennyLane Datasets — *What is a Quantum Dataset?*  
  https://pennylane.ai/datasets/what-is-a-quantum-dataset
- PennyLane Datasets — *Conventional Datasets for QML*  
  https://pennylane.ai/datasets/conventional-datasets-for-qml
- Complemento de catálogo / definición operativa:  
  https://pennylane.ai/blog/2023/08/introducing-pennylane-datasets/  
  (“Quantum data is any data that goes in or comes out of a quantum system.”)

**Desarrollo.**

1. *Qué es un quantum dataset (PennyLane).*  
   Una colección de datos que describe sistemas cuánticos y su evolución. En programación cuántica, esas *features* son tipicamente argumentos de entrada y salidas de una **quantum function** que define el sistema. Ejemplos explícitos en la página:
   - **Hamiltoniano** del sistema y otros observables.
   - **Estado** de interés (p. ej. ground state) y un circuito eficiente de **state preparation**.
   - **Unitarias** útiles (p. ej. operadores Clifford para *qubit tapering*).
   - **Medidas / proyectores**, distribuciones y valores esperados.
   - Parámetros de **evolución** y descripciones de **ruido** (p. ej. operadores de Kraus).

2. *Por qué catalogar datos cuánticos.* PennyLane motiva tres puntos:
   - Como en ML clásico, datasets accesibles y de calidad catalizan algoritmos nuevos y mejoras.
   - Aprender de datos cuánticos es un candidato más *intuitivo* a ventaja computacional cuántica que forzar datos clásicos en un circuito.
   - Datos listos reducen fricción interdisciplinar y aceleran trabajo reproducible.

3. *Qué son datos convencionales / clásicos para QML.*  
   Además de quantum data, los algoritmos cuánticos pueden trabajar con datos convencionales (“classical” data): imágenes, vídeo, audio, tablas etiquetadas, etc. La pregunta de investigación suele ser si un modelo QML puede etiquetar, categorizar o reconocer rasgos en esos objetos. Datasets estándar (Iris 1936, MNIST 1994, etc.) se usan como *benchmark*; muchos son demasiado grandes para modelos cuánticos actuales, así que se recurre a datasets históricos pequeños y/o a **reducción de dimensionalidad**.

4. *El cuello de botella no es solo el modelo: es el embedding.*  
   Con datos clásicos, un paso no trivial es **cómo embeber** la información en el sistema cuántico (angle / amplitude / basis embedding, feature maps, etc.). PennyLane lo trata como campo activo de investigación; los *embedding templates* automatizan parte del preproceso hacia vectores ortonormalizados u otras entradas compatibles con el circuito.

5. *Cómo encaja este TFM (mapa de carpetas).*
   - `Notebooks/ClassicalClassification/` (Iris): dataset **convencional** clásico, embebido / procesado para comparar baselines clásicos vs modelos cuánticos (QSVM, QNN, VQC). Encaja directamente en *Conventional Datasets for QML*.
   - `Notebooks/QuantumClassification/` (Bars and Stripes de PennyLane Datasets): dataset de **benchmark QML** alojado en el catálogo PennyLane (patrones imagen → vectores + etiquetas). Es útil como “dato de la literatura QML”, pero **no** es, en el sentido estricto de *What is a Quantum Dataset?*, un Hamiltoniano / estado / sombra clásica de un sistema físico. Hay que decirlo con cuidado en el documento: “dataset de clasificación usado en benchmarking QML (PennyLane)”, no “dato cuántico nativo del sistema”.
   - La comparación justa (mismo PCA a 4 componentes + mismo `MinMaxScaler[0, π]`) mide diferencia de **modelo / representación**, no diferencia de *naturaleza ontológica* del dato.

**Por verificar.**

- Confirmar en el paper de origen de *Bars and Stripes* en PennyLane (candidato: Bowles, Ahmed y Schuld, 2024, *Better than classical?*) cómo presentan el dataset: ¿synthetic classical patterns for QML benchmarks? ¿alguna interpretación como distribución de Born / QCBM?
- Revisar si en el documento conviene una tabla “tipo de dato × carpeta × qué se concluye”.
- Cita adicional opcional sobre embedding: glosario PennyLane *quantum embedding* y demos de Angle/Amplitude embedding (solo si se desarrolla el capítulo de codificación).

**Dónde se usa.** Documento: capítulo de contexto / datasets (antes de resultados Iris y Bars and Stripes). Defensa: diapositiva “qué es (y no es) un dataset cuántico” para anticipar la pregunta de rigor metodológico.

---

<!-- Siguiente argumento: copiar la plantilla de arriba -->

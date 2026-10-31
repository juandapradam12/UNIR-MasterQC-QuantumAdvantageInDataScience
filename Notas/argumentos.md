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

<!-- Siguiente argumento: copiar la plantilla de arriba -->

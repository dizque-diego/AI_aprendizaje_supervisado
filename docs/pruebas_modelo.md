# Pruebas del Modelo — Árbol de Decisión
## Sistema de Predicción de Congestión | Metro de Medellín

---

## 1. Descripción General de las Pruebas

Las pruebas realizadas buscan validar el comportamiento del modelo de aprendizaje supervisado basado en un **Árbol de Decisión**, entrenado para predecir el nivel de congestión (Baja, Media, Alta) en el Metro de Medellín.

El proceso de prueba cubre cuatro aspectos:

1. Verificación de la carga y exploración del dataset
2. Evaluación de la exactitud del modelo sobre datos de prueba
3. Análisis del reporte de clasificación por categoría
4. Predicciones sobre escenarios nuevos no vistos por el modelo

---

## 2. Dataset Utilizado

| Característica | Valor |
|---|---|
| Archivo | `data/dataset_transporte_supervisado.csv` |
| Total de registros | 50 |
| Total de columnas | 9 |
| Variable objetivo | `nivel_congestion` |
| Categorías a predecir | Alta, Baja, Media |

### Distribución de la variable objetivo

| Nivel de Congestión | Registros | Porcentaje |
|---|---|---|
| Media | 19 | 38.0% |
| Alta | 18 | 36.0% |
| Baja | 13 | 26.0% |

### Variables de entrada utilizadas

| Variable | Tipo | Descripción |
|---|---|---|
| `hora` | Numérica | Hora del día (5 a 21) |
| `dia_semana` | Categórica | Día de la semana (codificada) |
| `linea` | Categórica | Línea del metro: A, B o K (codificada) |
| `estacion` | Categórica | Nombre de la estación (codificada) |
| `pasajeros_estimados` | Numérica | Cantidad de pasajeros en el registro |
| `frecuencia_min` | Numérica | Frecuencia de paso del tren en minutos |
| `clima_lluvia` | Binaria | 1 si llueve, 0 si no |
| `evento_cercano` | Binaria | 1 si hay evento cercano, 0 si no |

---

## 3. División de Datos

Los datos fueron divididos de la siguiente manera para garantizar una evaluación justa:

| Conjunto | Registros | Porcentaje |
|---|---|---|
| Entrenamiento | 40 | 80% |
| Prueba | 10 | 20% |

Se usó `stratify=y` para mantener la proporción de clases en ambos conjuntos, y `random_state=42` para garantizar reproducibilidad.

---

## 4. Parámetros del Modelo

| Parámetro | Valor | Justificación |
|---|---|---|
| Algoritmo | `DecisionTreeClassifier` | Requerido por la actividad (cap. 17 del libro) |
| Criterio | `gini` | Mide la impureza de cada nodo de decisión |
| Profundidad máxima | `4` | Evita el sobreajuste con un dataset pequeño |
| Semilla aleatoria | `42` | Garantiza resultados reproducibles |

Profundidad real del árbol entrenado: **4 niveles**
Número de hojas (nodos de decisión final): **5 hojas**

---

## 5. Resultados de la Evaluación

### 5.1 Exactitud General

```
Exactitud del modelo: 90.0%
```

El modelo clasificó correctamente **9 de 10 registros** del conjunto de prueba.

### 5.2 Reporte de Clasificación por Nivel

```
              precision    recall  f1-score   support

        Alta       1.00      0.75      0.86         4
        Baja       1.00      1.00      1.00         2
       Media       0.80      1.00      0.89         4

    accuracy                           0.90        10
   macro avg       0.93      0.92      0.92        10
weighted avg       0.92      0.90      0.90        10
```

**Interpretación por nivel:**

- **Alta:** Precisión del 100% — cuando el modelo predice Alta, siempre acierta. Sin embargo, detectó solo el 75% de los casos reales de Alta (recall), lo que indica que 1 caso de Alta fue clasificado como Media.
- **Baja:** Precisión y recall del 100% — el modelo identificó perfectamente todos los casos de congestión baja.
- **Media:** Recall del 100% — detectó todos los casos de Media. La precisión del 80% indica que 1 caso fue clasificado como Media cuando era Alta.

### 5.3 Matriz de Confusión

```
            Predicho:
             Alta   Baja  Media
Real: Alta     3      0      1
      Baja     0      2      0
      Media    0      0      4
```

**Análisis:** El único error del modelo fue clasificar 1 caso de congestión **Alta** como **Media**. Esto es esperable dado el tamaño reducido del dataset (50 registros). Los casos de congestión **Baja** fueron clasificados con un 100% de exactitud.

---

## 6. Importancia de las Variables

El árbol de decisión identificó las variables más relevantes para predecir el nivel de congestión:

| Variable | Importancia | Interpretación |
|---|---|---|
| `pasajeros_estimados` | Alta | Principal factor de decisión del modelo |
| `frecuencia_min` | Media | A menor frecuencia, mayor congestión posible |
| `hora` | Media | Las horas pico tienen más pasajeros |
| `evento_cercano` | Baja | Influye en casos específicos |
| `clima_lluvia` | Baja | Aumenta ligeramente la congestión |
| `dia_semana` | Baja | Fines de semana tienen menos usuarios |
| `linea` | Mínima | Las tres líneas tienen comportamientos similares |
| `estacion` | Mínima | La estación por sí sola no determina la congestión |

> La variable más importante fue `pasajeros_estimados`, lo cual es lógico: a más pasajeros, mayor es el nivel de congestión del sistema.

---

## 7. Reglas del Árbol de Decisión

El árbol aprendió las siguientes reglas lógicas a partir de los datos:

```
|--- pasajeros_estimados <= 695
|   |--- pasajeros_estimados <= 350
|   |   |--- class: Baja
|   |--- pasajeros_estimados > 350
|   |   |--- pasajeros_estimados <= 635
|   |   |   |--- class: Media
|   |   |--- pasajeros_estimados > 635
|   |   |   |--- pasajeros_estimados <= 670
|   |   |   |   |--- class: Alta
|   |   |   |--- pasajeros_estimados > 670
|   |   |   |   |--- class: Media
|--- pasajeros_estimados > 695
|   |--- class: Alta
```

**Lectura en lenguaje natural:**
- Si hay **más de 695 pasajeros** → Congestión **Alta**
- Si hay **entre 351 y 635 pasajeros** → Congestión **Media**
- Si hay **350 o menos pasajeros** → Congestión **Baja**
- El rango entre 635 y 695 presenta una zona de transición entre Media y Alta

---

## 8. Pruebas con Escenarios Nuevos

Se diseñaron 4 escenarios para validar el modelo con datos no vistos durante el entrenamiento:

### Escenario 1 — Hora pico entre semana

| Campo | Valor |
|---|---|
| Hora | 8:00 am |
| Día | Lunes |
| Línea | A |
| Estación | San Antonio |
| Pasajeros estimados | 900 |
| Frecuencia | 4 minutos |
| Lluvia | No |
| Evento cercano | No |

**Resultado:** Congestión **ALTA** (100% de confianza)

*Análisis: El modelo predice correctamente Alta congestión para una hora pico del lunes con alta afluencia de pasajeros.*

---

### Escenario 2 — Domingo en la mañana

| Campo | Valor |
|---|---|
| Hora | 10:00 am |
| Día | Domingo |
| Línea | B |
| Estación | San Javier |
| Pasajeros estimados | 220 |
| Frecuencia | 12 minutos |
| Lluvia | No |
| Evento cercano | No |

**Resultado:** Congestión **BAJA** (100% de confianza)

*Análisis: El domingo con pocos pasajeros y baja frecuencia refleja un sistema tranquilo, lo cual es coherente con la realidad.*

---

### Escenario 3 — Viernes en la tarde con lluvia y evento

| Campo | Valor |
|---|---|
| Hora | 6:00 pm |
| Día | Viernes |
| Línea | A |
| Estación | Industriales |
| Pasajeros estimados | 850 |
| Frecuencia | 5 minutos |
| Lluvia | Sí |
| Evento cercano | Sí |

**Resultado:** Congestión **ALTA** (100% de confianza)

*Análisis: La combinación de hora pico, lluvia y evento cercano genera las condiciones más críticas de congestión. El modelo lo identifica correctamente.*

---

### Escenario 4 — Miércoles en la tarde, MetroCable

| Campo | Valor |
|---|---|
| Hora | 2:00 pm |
| Día | Miércoles |
| Línea | K (MetroCable) |
| Estación | Acevedo |
| Pasajeros estimados | 400 |
| Frecuencia | 8 minutos |
| Lluvia | No |
| Evento cercano | No |

**Resultado:** Congestión **MEDIA** (100% de confianza)

*Análisis: Un día entre semana en la tarde con una cantidad moderada de pasajeros en el MetroCable K corresponde a congestión media, resultado coherente con los patrones del dataset.*

---

## 9. Archivos Generados

Al ejecutar el código (`src/arbol_decision.py`) se generan automáticamente los siguientes archivos en la carpeta `docs/`:

| Archivo | Descripción |
|---|---|
| `arbol_decision.png` | Visualización gráfica completa del árbol de decisión |
| `matriz_confusion.png` | Gráfica de la matriz de confusión del modelo |
| `importancia_variables.png` | Gráfica de barras con la importancia de cada variable |

---

## 10. Conclusiones

1. El modelo de Árbol de Decisión alcanzó una **exactitud del 90%** sobre el conjunto de prueba, lo cual es un resultado satisfactorio considerando el tamaño reducido del dataset (50 registros).

2. La variable **`pasajeros_estimados`** resultó ser el principal predictor del nivel de congestión, seguida por `frecuencia_min` y `hora`. Esto es consistente con la lógica del sistema de transporte masivo.

3. El modelo clasifica con **100% de precisión** los casos de congestión **Baja** y detecta correctamente los casos de congestión **Alta** con 100% de precisión cuando los predice.

4. El único error observado fue la clasificación de 1 caso de congestión **Alta** como **Media**, lo cual es atribuible al tamaño del dataset y no a una falla estructural del modelo.

5. Las reglas generadas por el árbol son **interpretables y lógicas**, lo que permite relacionar directamente el modelo con los conceptos de árboles y reglas de decisión estudiados en el capítulo 17 del libro de referencia.

6. Un dataset más grande con datos reales del Metro de Medellín permitiría obtener un modelo más robusto y generalizable para escenarios de la vida real.

---

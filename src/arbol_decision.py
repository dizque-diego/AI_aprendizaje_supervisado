# =============================================================
# SISTEMA DE CLASIFICACION DE CONGESTION - METRO DE MEDELLIN
# Modelo: Arbol de Decision (Aprendizaje Supervisado)
# Materia: Inteligencia Artificial
# =============================================================
#
# Este archivo implementa un modelo de aprendizaje supervisado
# basado en un Arbol de Decision para predecir el nivel de
# congestion (Baja, Media, Alta) en el Metro de Medellin.
#
# Pasos del proceso:
#   1. Cargar el dataset desde el archivo CSV
#   2. Explorar y entender los datos
#   3. Preprocesar los datos (convertir texto a numeros)
#   4. Dividir en conjunto de entrenamiento y prueba
#   5. Entrenar el modelo Arbol de Decision
#   6. Evaluar el modelo con metricas
#   7. Realizar predicciones con nuevos datos
#   8. Visualizar el arbol de decision

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# PASO 1: CARGAR EL DATASET
# ─────────────────────────────────────────────

print("=" * 60)
print("  SISTEMA DE PREDICCION DE CONGESTION")
print("  Metro de Medellin — Arbol de Decision")
print("=" * 60)

print("\n[1] Cargando dataset...")

df = pd.read_csv("../data/dataset_transporte_supervisado.csv")
print(f"    Dataset cargado: {df.shape[0]} registros, {df.shape[1]} columnas")


# ─────────────────────────────────────────────
# PASO 2: EXPLORAR LOS DATOS
# ─────────────────────────────────────────────

print("\n[2] Explorando los datos...")
print("\n    Primeras 5 filas del dataset:")
print(df.head().to_string(index=False))

print("\n    Distribucion de la variable objetivo (nivel_congestion):")
conteo = df['nivel_congestion'].value_counts()
for nivel, cantidad in conteo.items():
    porcentaje = (cantidad / len(df)) * 100
    print(f"      {nivel:<8}: {cantidad} registros ({porcentaje:.1f}%)")

print("\n    Estadisticas de variables numericas:")
print(df[['hora', 'pasajeros_estimados', 'frecuencia_min']].describe().to_string())


# ─────────────────────────────────────────────
# PASO 3: PREPROCESAR LOS DATOS
# ─────────────────────────────────────────────

print("\n[3] Preprocesando datos...")

df_modelo = df.copy()

encoder_dia      = LabelEncoder()
encoder_linea    = LabelEncoder()
encoder_estacion = LabelEncoder()
encoder_objetivo = LabelEncoder()

df_modelo['dia_semana'] = encoder_dia.fit_transform(df_modelo['dia_semana'])
df_modelo['linea']      = encoder_linea.fit_transform(df_modelo['linea'])
df_modelo['estacion']   = encoder_estacion.fit_transform(df_modelo['estacion'])

print("    Codificacion de dias de la semana:")
for codigo, nombre in enumerate(encoder_dia.classes_):
    print(f"      {nombre} = {codigo}")

print("    Codificacion de lineas:")
for codigo, nombre in enumerate(encoder_linea.classes_):
    print(f"      Linea {nombre} = {codigo}")

columnas_entrada = [
    'hora', 'dia_semana', 'linea', 'estacion',
    'pasajeros_estimados', 'frecuencia_min',
    'clima_lluvia', 'evento_cercano'
]

X = df_modelo[columnas_entrada]
y = encoder_objetivo.fit_transform(df_modelo['nivel_congestion'])

print(f"\n    Variables de entrada (X): {list(columnas_entrada)}")
print(f"    Codificacion objetivo: {list(encoder_objetivo.classes_)}")
print(f"      (0={encoder_objetivo.classes_[0]}, "
      f"1={encoder_objetivo.classes_[1]}, "
      f"2={encoder_objetivo.classes_[2]})")


# ─────────────────────────────────────────────
# PASO 4: DIVIDIR EN ENTRENAMIENTO Y PRUEBA
# ─────────────────────────────────────────────

print("\n[4] Dividiendo datos en entrenamiento y prueba...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"    Total de registros:       {len(X)}")
print(f"    Registros entrenamiento:  {len(X_train)} (80%)")
print(f"    Registros prueba:         {len(X_test)} (20%)")


# ─────────────────────────────────────────────
# PASO 5: ENTRENAR EL ARBOL DE DECISION
# ─────────────────────────────────────────────

print("\n[5] Entrenando el Arbol de Decision...")

modelo = DecisionTreeClassifier(
    max_depth=4,
    criterion='gini',
    random_state=42
)

modelo.fit(X_train, y_train)

print("    Modelo entrenado correctamente.")
print(f"    Profundidad del arbol: {modelo.get_depth()}")
print(f"    Numero de hojas:       {modelo.get_n_leaves()}")


# ─────────────────────────────────────────────
# PASO 6: EVALUAR EL MODELO
# ─────────────────────────────────────────────

print("\n[6] Evaluando el modelo...")

y_pred    = modelo.predict(X_test)
exactitud = accuracy_score(y_test, y_pred)

print(f"\n    Exactitud del modelo: {exactitud * 100:.1f}%")

print("\n    Reporte de clasificacion por nivel:")
print("    " + "-" * 52)
reporte = classification_report(
    y_test, y_pred,
    target_names=encoder_objetivo.classes_,
    zero_division=0
)
for linea_rep in reporte.split('\n'):
    print("    " + linea_rep)

cm = confusion_matrix(y_test, y_pred)
clases = encoder_objetivo.classes_
print("    Matriz de confusion:")
print(f"    {'':>10}", end="")
for c in clases:
    print(f"  {c:>6}", end="")
print()
for i, fila in enumerate(cm):
    print(f"    {clases[i]:>10}", end="")
    for val in fila:
        print(f"  {val:>6}", end="")
    print()


# ─────────────────────────────────────────────
# PASO 7: PREDICCIONES CON NUEVOS ESCENARIOS
# ─────────────────────────────────────────────

print("\n[7] Predicciones con nuevos escenarios...")
print("    " + "-" * 56)

escenarios = [
    {
        "descripcion": "Lunes 8am, Linea A, San Antonio, hora pico",
        "hora": 8, "dia_semana": "Lunes", "linea": "A",
        "estacion": "San Antonio", "pasajeros_estimados": 900,
        "frecuencia_min": 4, "clima_lluvia": 0, "evento_cercano": 0,
    },
    {
        "descripcion": "Domingo 10am, Linea B, San Javier, dia tranquilo",
        "hora": 10, "dia_semana": "Domingo", "linea": "B",
        "estacion": "San Javier", "pasajeros_estimados": 220,
        "frecuencia_min": 12, "clima_lluvia": 0, "evento_cercano": 0,
    },
    {
        "descripcion": "Viernes 6pm, Linea A, Industriales, lluvia y evento",
        "hora": 18, "dia_semana": "Viernes", "linea": "A",
        "estacion": "Industriales", "pasajeros_estimados": 850,
        "frecuencia_min": 5, "clima_lluvia": 1, "evento_cercano": 1,
    },
    {
        "descripcion": "Miercoles 2pm, Linea K, Acevedo, tarde normal",
        "hora": 14, "dia_semana": "Miercoles", "linea": "K",
        "estacion": "Acevedo", "pasajeros_estimados": 400,
        "frecuencia_min": 8, "clima_lluvia": 0, "evento_cercano": 0,
    },
]

for i, esc in enumerate(escenarios, 1):
    fila = pd.DataFrame([{
        'hora':                esc['hora'],
        'dia_semana':          encoder_dia.transform([esc['dia_semana']])[0],
        'linea':               encoder_linea.transform([esc['linea']])[0],
        'estacion':            encoder_estacion.transform([esc['estacion']])[0],
        'pasajeros_estimados': esc['pasajeros_estimados'],
        'frecuencia_min':      esc['frecuencia_min'],
        'clima_lluvia':        esc['clima_lluvia'],
        'evento_cercano':      esc['evento_cercano'],
    }])

    prediccion_num = modelo.predict(fila)[0]
    prediccion     = encoder_objetivo.inverse_transform([prediccion_num])[0]
    probabilidades = modelo.predict_proba(fila)[0]

    print(f"\n    Escenario {i}: {esc['descripcion']}")
    print(f"    Prediccion: CONGESTION {prediccion.upper()}")
    for j, clase in enumerate(encoder_objetivo.classes_):
        barra = "█" * int(probabilidades[j] * 20)
        print(f"      {clase:<6}: {probabilidades[j]*100:5.1f}%  {barra}")

print("\n    " + "-" * 56)


# ─────────────────────────────────────────────
# PASO 8: VISUALIZACIONES
# ─────────────────────────────────────────────

print("\n[8] Generando visualizaciones...")

# Arbol de decision grafico
fig1, ax1 = plt.subplots(figsize=(20, 10))
plot_tree(
    modelo,
    feature_names=columnas_entrada,
    class_names=encoder_objetivo.classes_,
    filled=True, rounded=True, fontsize=9, ax=ax1
)
ax1.set_title(
    "Arbol de Decision — Prediccion de Congestion\nMetro de Medellin",
    fontsize=14, fontweight='bold', pad=20
)
plt.tight_layout()
plt.savefig("../docs/arbol_decision.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Guardado: docs/arbol_decision.png")

# Matriz de confusion
fig2, ax2 = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases)
disp.plot(ax=ax2, colorbar=False, cmap='Blues')
ax2.set_title(
    "Matriz de Confusion\nNivel de Congestion Metro Medellin",
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
plt.savefig("../docs/matriz_confusion.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Guardado: docs/matriz_confusion.png")

# Importancia de variables
importancias = modelo.feature_importances_
indices = importancias.argsort()[::-1]
nombres = [columnas_entrada[i] for i in indices]
valores = [importancias[i] for i in indices]

fig3, ax3 = plt.subplots(figsize=(9, 5))
colores = ['#1565C0' if v == max(valores) else '#90CAF9' for v in valores]
bars = ax3.barh(nombres[::-1], valores[::-1], color=colores[::-1])
ax3.set_xlabel("Importancia (Gini)", fontsize=11)
ax3.set_title(
    "Importancia de Variables — Arbol de Decision\nMetro de Medellin",
    fontsize=12, fontweight='bold'
)
for bar, val in zip(bars, valores[::-1]):
    ax3.text(
        bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
        f'{val:.3f}', va='center', fontsize=9
    )
ax3.set_xlim(0, max(valores) + 0.1)
plt.tight_layout()
plt.savefig("../docs/importancia_variables.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Guardado: docs/importancia_variables.png")

# Reglas del arbol en texto
print("\n[9] Reglas del arbol (primeras 30 lineas):")
reglas = export_text(modelo, feature_names=list(columnas_entrada))
for linea_regla in reglas.split('\n')[:30]:
    print("    " + linea_regla)
print("    ...")

print("\n" + "=" * 60)
print("  PROCESO COMPLETADO EXITOSAMENTE")
print(f"  Exactitud final del modelo: {exactitud * 100:.1f}%")
print("  Archivos generados en docs/:")
print("    - arbol_decision.png")
print("    - matriz_confusion.png")
print("    - importancia_variables.png")
print("=" * 60)
 

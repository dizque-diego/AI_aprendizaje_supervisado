## Aprendizaje supervisado aplicado al Metro de Medellín

Este proyecto desarrolla un modelo de aprendizaje supervisado tomando como referencia la operación del sistema de transporte masivo Metro de Medellín.

El proyecto toma como referencia algunas características generales del Metro de Medellín tales como líneas, estaciones y tiempos aproximados entre trenes, sin embargo no se cuenta con una fuente oficial de datos operativos para todas las variables requeridas, por lo tanto se construye un dataset simulado para fines académicos.

# Objetivo Principal

Clasificar el nivel de congestión del sistema a partir de diferentes indicadores como:

* Hora del día
* Día de la semana
* Línea del sistema
* Cantidad de pasajeros estimados
* Frecuencia del servicio
* Clima y otros posibles eventos

# Problema a resolver:

Predecir el nivel de congestión en una línea o componente determinado del sistema.

# Pregunta problematizadora:

¿Qué nivel de congestión puede esperarse en una línea o estación del Metro según el día de la semana, la hora, la cantidad de pasajeros, la frecuencia del servicio y otros eventos como el clima?

# Tipo de Aprendizaje:

Se utiliza el aprendizaje supervisado, ya que el dataset cuenta con la variable `nivel_congestion`.

Esta variable representa las categorías que el modelo debe aprender a predecir, entre las cuales están:

* Baja
* Media
* Alta

A partir de registros previamente clasificados, el modelo puede aprender patrones relacionados con las condiciones de la operación del sistema y realizar predicciones en nuevos casos.

## Modelo propuesto

Se propone un árbol de decisión, implementado en Python mediante la librería `scikit-learn`.

Debido a que facilita resolver problemas de clasificación y generar reglas de decisión interpretables, facilitando comprender cómo el modelo clasifica cada registro y permitiendo relacionar el presente ejercicio con los conceptos de árboles y reglas de decisión.

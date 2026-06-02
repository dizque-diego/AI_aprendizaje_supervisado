## Descripción de los datos

El dataset 'dataset_transporte_supervisado.csv' fue creado con fines académicos para desarrollar un modelo de aprendizaje de análisis de congestión en un sistema de transporte masivo.

El conjunto de datos se basa en algunas características generales del Metro de Medellín, en ningún sentido se toman datos oficiales.

Las variables que se incluyen en el dataset son:
- hora
- dia_semana
- linea
- estacion
- pasajeros_estimados
- frecuencia_min
- clima_lluvia
- evento_cercano
- nivel_congestion

# Variable objetivo

Se toma la variable 'nivel_congestion' como la variable que el modelo debe aprender a predecir, clasificando cada registro en una de las siguientes categorías:
- Baja
- Media
- Alta

# Limitaciones

El dataset, al ser creado con fines académicos, presenta una determinada variedad de limitaciones tales como:
- No cuenta con registros reales ni históricos de operación
- Cuenta con un tamaño reducido

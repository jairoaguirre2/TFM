## **Predicción mediante machine learning de mutantes enzimáticos para la degradación de plásticos: aplicación a cutinasas**

Este repositorio contiene el código desarrollado para el Trabajo Fin de Máster 

El trabajo está orientado al análisis y la predicción de enzimas mutantes mediante técnicas 
de *Machine Learning* y bioinformática.

El proyecto implementa un *pipeline* completo de entrenamiento y predicción, que incluye
la codificación de secuencias, el entrenamiento de modelos supervisados y la inferencia
sobre nuevos mutantes.

## Estructura del repositorio

* Pipeline: Contiene el flujo final del trabajo y consta de cuatro archivos:
  1. funciones.py: Funciones generales para codificación de secuencias, definición de autoencoder y mutaciones
  2. modelos.py: Funciones para preparar datos, entrenar y optimizar modelos, evaluación de métricas y generar
     tablas y gráficas
  3. entrnemiento.py: *Script* principal para el entrenamiento de modelos, selección de los mejores enfoques
     y reentrenamiento con los hiperparámetros óptimos
  4. predicciones.py: Inferencia masiva sobre librerías combinatorias de mutantes y análisis posterior
     de los resultados.

* Pruebas: Contiene *scripts* utilizados durante la fase exploratoria del trabajo:
  1. autoencoder.py: *Pipeline* alternativo basado en autoencoder para aprendizaje no supervisado de
     secuencias proteicas.
  2. modelos_preeliminares.py: Modelos y análisis exploratorios iniciales usando datos experimentales
  3. test_train.py: Modelos y análisis exploratorios iniciales usando datos públicos
 
* Datos: Incluye los conjuntos de datos utilizados en el proyecto:
  1. EC_3_reviewed_200_400.fasta.gz: Secuencias de hidrolasas descargadas de Uniprot, filtradas por longitud
  2. EnzymesMutationsData.zip: Datos públicos descargados de Kaggle, donde se usó *train_dataset.csv* como
     dataset principal para el entrenamiento y evaluación de modelos.




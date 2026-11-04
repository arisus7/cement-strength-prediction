# Predicción de Resistencia de Cemento (GCC) - Datathon 2026

Este repositorio contiene la implementación técnica de un pipeline de Machine Learning de punta a punta diseñado para predecir la resistencia a la compresión del cemento a los 1, 7 y 28 días de curado. El proyecto fue desarrollado íntegramente en Databricks utilizando la arquitectura de datos Medallón y Unity Catalog.

## Arquitectura del Proyecto (Medallón)

El pipeline está estructurado en cinco etapas secuenciales para garantizar la calidad de los datos y la reproducibilidad del modelo:

*   01_bronze: Ingesta de datos crudos desde el volumen de Unity Catalog.
*   02_silver: Limpieza de datos, imputación de valores nulos y estandarización de tipos de datos.
*   03_gold: Ingeniería de características (Feature Engineering). Incluye la codificación de variables categóricas (evitando la trampa de variables dummy mediante la exclusión de `pt_other`) y la resolución de redundancias binarias.
*   03b_eda_vif: Análisis Exploratorio de Datos y validación estadística. Se aplicó el Factor de Inflación de Varianza (VIF), resultando en la eliminación de `Mineral_Phase_C` para corregir la multicolinealidad composicional y aportar estabilidad matemática al algoritmo.
*   04_train: Entrenamiento de modelos por horizonte temporal, validación estricta con GroupKFold, registro de métricas y artefactos en MLflow, y empaquetado del modelo final mediante la interfaz unificada de MLflow PyFunc.

## Resultados y Modelo Campeón

*   Algoritmo seleccionado: ExtraTrees.
*   Precisión: Errores porcentuales (MAPE) entre 4.08% y 4.49% en el conjunto de prueba (datos no vistos).
*   Validación Física: La estructura de predicción garantiza la monotonía física de la mezcla (Resistencia 1 día <= Resistencia 7 días <= Resistencia 28 días).
*   Análisis de Residuales: La distribución del error demuestra un comportamiento conservador (ligero sesgo positivo). En los casos atípicos, el modelo tiende a subestimar la resistencia real, mitigando el riesgo operativo para el negocio de la construcción.

## Instrucciones de Ejecución (Reproducibilidad)

Para ejecutar este pipeline desde cero en un entorno de Databricks, siga las instrucciones a continuación.

### 1. Preparación del Entorno (Unity Catalog)
Ejecute el siguiente código SQL en su workspace para crear la jerarquía de almacenamiento:

```sql
CREATE CATALOG IF NOT EXISTS datathon;
CREATE SCHEMA IF NOT EXISTS datathon.gcc;
CREATE VOLUME IF NOT EXISTS datathon.gcc.raw;

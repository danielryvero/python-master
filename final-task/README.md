# 📊 Loan Approval Prediction

## 📝 Enunciado

### 🎯 Objetivo
Construir un flujo de trabajo completo de análisis y modelado para predecir si un cliente califica para un préstamo bancario. El problema se plantea como una **clasificación binaria**:  
- `1` → **Aprobado**  
- `0` → **No Aprobado**

Este caso práctico permite aplicar conocimientos clave de ciencia de datos: desde manipulación de datos en Python hasta la creación y evaluación de modelos de machine learning.

---

## 🧠 Contexto

Imagina que trabajas en un banco y necesitas desarrollar un modelo para decidir si otorgar un préstamo a un cliente. Utilizarás datos simulados para entrenar un modelo predictivo que clasifique a los clientes en función de su perfil financiero.

---

## 📁 Datos

Se ha generado un conjunto de datos sintético con al menos **50 registros** en el archivo `datos_prestamos.csv`. Las columnas incluidas son:

- `Edad del cliente`  
- `Ingresos mensuales del cliente`  
- `Puntuación crediticia` (entre 300 y 850)  
- `Número de préstamos previos`  
- `Categoría de préstamo` (`0` = No Aprobado, `1` = Aprobado)

---

## 🧪 Flujo de Trabajo

1. **Cargar y explorar los datos**
2. **Preprocesar**: normalización de variables numéricas y codificación de variables categóricas (si las hubiera)
3. **División** del conjunto de datos en entrenamiento y prueba
4. **Construcción del modelo** de clasificación (ej. Regresión Logística con `scikit-learn`)
5. **Entrenamiento** del modelo
6. **Evaluación** de rendimiento (accuracy, precision, recall, etc.)
7. **Predicciones** con nuevos datos de clientes

---

## 🧰 Herramientas y Librerías

- Python (Pandas, NumPy)
- Scikit-learn
- Matplotlib / Seaborn (visualización)

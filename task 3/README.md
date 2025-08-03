# 🛒 Product Purchase Prediction

## 📝 Enunciado

### 🎯 Objetivo

Desarrollar un modelo de Machine Learning capaz de predecir si un cliente potencial realizará la compra de un producto, basándose en características como su edad, historial de compras y tiempo de navegación en el sitio web. Este modelo ayudará a personalizar campañas de marketing y mejorar la conversión de clientes.

---

## 🧠 Contexto

Una empresa de comercio electrónico busca optimizar su estrategia de ventas anticipando el comportamiento de sus clientes. Con información del comportamiento online y el historial de compras, se pretende construir un clasificador binario que determine la probabilidad de que un usuario realice una compra.

---

## 📁 Datos

La base de datos contiene información de **100 clientes potenciales**, recopilada en el archivo `Estudiodecaso6.csv`.

### 📊 Columnas incluidas:

- `Edad` (en años)  
- `Compras anteriores` (número de compras realizadas previamente)  
- `Tiempo en el sitio web` (en minutos)  
- `Compra realizada` (`0` = No compró, `1` = Compró)

---

## 🔄 Flujo de Trabajo

1. **Adquisición y preparación de los datos**  
   - Cargar el archivo `Estudiodecaso6.csv`
   - Verificar y tratar valores nulos, tipos de datos y escalado si es necesario

2. **Análisis exploratorio de datos (EDA)**  
   - Analizar la distribución de las variables
   - Explorar correlaciones con la variable objetivo (`Compra realizada`)
   - Visualizar relaciones relevantes

3. **Entrenamiento del modelo**  
   - Dividir los datos en conjuntos de entrenamiento y prueba  
   - Elegir un algoritmo de clasificación (e.g., Regresión Logística, KNN, Random Forest)  
   - Entrenar el modelo

4. **Realización de predicciones**  
   - Predecir la compra de nuevos clientes potenciales usando el modelo entrenado

5. **Evaluación del rendimiento**  
   - Utilizar métricas como **precisión**, **recall**, y **F1-score**

6. **Mejora y optimización del modelo**  
   - Ajuste de hiperparámetros  
   - Selección de características  
   - Prueba con diferentes algoritmos

---

## 🧰 Herramientas y Librerías

- Python (Pandas, NumPy, Scikit-learn)
- Matplotlib / Seaborn (para visualización)
- Jupyter Notebook / Google Colab


# 4_modelo_logistico.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Cargar los datos
df = pd.read_csv("datos_prestamos.csv")

# Preprocesamiento (si es necesario)
# Aseguramos que las columnas numéricas sean correctamente tratadas
df['Edad'] = df['Edad'].astype(float)
df['Ingresos'] = df['Ingresos'].astype(float)
df['Puntuacion_Credito'] = df['Puntuacion_Credito'].astype(float)
df['Prestamos_Previos'] = df['Prestamos_Previos'].astype(float)

# Dividir los datos
X = df[['Edad', 'Ingresos', 'Puntuacion_Credito', 'Prestamos_Previos']]
y = df['Aprobado']

# Normalización
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Crear y entrenar el modelo
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluar el modelo
y_pred = model.predict(X_test)

# 1. Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print("Matriz de Confusión:")
print(cm)

# 2. Precisión, Recall y F1-Score. Evalua cómo funciona el modelo en cuanto a balance de clases
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# 3. Curva ROC y AUC. 
# La curva ROC ayuda a evaluar la capacidad del modelo de para discriminar entre las clases
# El área bajo la curva o AUC se utiliza como métrica para indicar la calidad de las predicciones
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de falsos positivos')
plt.ylabel('Tasa de verdaderos positivos')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()

# Reflexión sobre los falsos positivos y negativos
print("\nReflexión sobre los Falsos Positivos y Negativos:")
print("""
Los falsos positivos como cuando un cliente no aprobado es clasificado como aprobado, podrían llevar a
que el banco otorgue un préstamo a un cliente que no sea capaz de devolverlo, aumentando el riesgo
de impago.  Los falsos negativos, como cuando un cliente aprobado es clasificado como no
aprobado, podrían generar oportunidades perdidas, ya que el banco rechazaría a un cliente que podría
haber cumplido con las condiciones del préstamo.
""")

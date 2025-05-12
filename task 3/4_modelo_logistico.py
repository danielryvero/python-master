import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# cargamos los datos
df = pd.read_csv('datos_prestamos.csv')

# separar caracteristicas y variable objetivo
X = df.drop('Aprobado', axis=1)
y = df['Aprobado']

# dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# crear y entrenar modelo de regresion logistica
modelo = LogisticRegression()
modelo.fit(X_train, y_train)

# hacer predicciones sobre el conjunto de prueba
y_pred = modelo.predict(X_test)

# evaluar modelo 
print("Precision: ", accuracy_score(y_test, y_pred))
print("Reporte de clasificacion:\n", classification_report(y_test, y_pred))
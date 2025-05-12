import tensorflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import pickle
import json
from sklearn.utils import class_weight
import numpy as np

# cargamos los datos almacenados en la  misma carpeta que el environment creado
df = pd.read_csv("Estudiodecaso6.csv", sep=';')

# preparar datos, dividiendo los datos según el parámetro a predecir
X = df.drop("Compra realizada", axis=1)
y = df["Compra realizada"]


# dividimos datos para entrenar el modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# estandarizar los datos transformar los datos con fit
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# crear y entrenar el modelo
model = Sequential([
    Dense(16, activation='relu', input_shape=(X.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=4, verbose=1)



# Calcular los pesos según la distribución de las clases
pesos = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

# Convertir a diccionario {0: peso_0, 1: peso_1}
pesos_dict = dict(enumerate(pesos))
print("Pesos aplicados por clase:", pesos_dict)

# Entrenar el modelo con los pesos de clase
history = model.fit(
    X_train_scaled,
    y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test_scaled, y_test),
    class_weight=pesos_dict,
    verbose=1
)

# evaluar
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nPrecisión en el conjunto de prueba: {accuracy:.2f}")

# predicciones y métricas
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))



###### conclusiones de las mejoras ######
# Mejor accuracy, F1 global y recall en la clase 1.
# El balanceo ayudó al modelo a enfocarse más en los clientes que sí compran (clase mayoritaria).
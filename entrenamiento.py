# ---------------------------
# 2. ENTRENAMIENTO DEL MODELO (Keras)
# ---------------------------
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

df = pd.read_csv("Estudiodecaso6.csv", sep=';')

# Preparar datos
X = df.drop("Compra realizada", axis=1)
y = df["Compra realizada"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Crear y entrenar el modelo
model = Sequential([
    Dense(16, activation='relu', input_shape=(X.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=4, verbose=1)

# Evaluar
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nPrecisión en el conjunto de prueba: {accuracy:.2f}")

# Predicciones y métricas
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))

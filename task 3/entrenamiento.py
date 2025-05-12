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

# elegimos el algoritmo de optimizacion y funcion de perdida como los usados en clase
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=4, verbose=1)

# crear historia para la evaluacion
history = model.fit(
    X_train_scaled,
    y_train,
    epochs=50,
    batch_size=8,
    validation_data=(X_test_scaled, y_test)
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

# salvemos el modelo para usarlo en otro archivo
keras.saving.save_model(model, 'modelo_entrenado.keras')
#model.save("modelo_entrenado.keras")

# salvemos el historial para la evaluacion luego
model.save("modelo_compras.h5")

# guardar el scaler
with open ("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
    
# guardado del history
with open("history.json", "w") as f:
    json.dump(history.history, f)
import pickle
import json
from keras.models import load_model
import matplotlib.pyplot as plt

# 1. Cargar el modelo entrenado
model = load_model("modelo_compras.h5")

# 2. Cargar el scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# 3. Cargar el historial de entrenamiento
with open("history.json", "r") as f:
    history = json.load(f)

# 4. Graficar accuracy y loss por época
plt.figure(figsize=(12, 5))

# precision
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Accuracy')
plt.plot(history['val_accuracy'], label='Val Accuracy')
plt.xlabel("Épocas")
plt.ylabel("Precisión")
plt.title("Precisión del modelo")
plt.legend()

# perdida
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.title("Pérdida del modelo")
plt.legend()

plt.tight_layout()
plt.show()

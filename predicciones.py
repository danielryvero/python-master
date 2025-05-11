from tensorflow import keras
import numpy as np

# carga el modelo guardado
model = keras.models.load_model("modelo_entrenado.keras")

# datos de nuevos clientes
nuevos_clientes = np.array([
    [30, 2, 40],
    [50, 1, 15],
    [33, 3, 50],   
    [29, 2, 45],   
    [52, 1, 10],   
    [35, 4, 70],   
    [23, 0, 20],   
    [21, 0, 5]     
])

# hacer predicciones
predicciones = model.predict(nuevos_clientes)

# mostrar resultados
for i, pred in enumerate(predicciones):
    resultado = "Sí comprará" if pred > 0.5 else "No comprará"
    print(f"Cliente {i+1}: {resultado} (Probabilidad: {pred[0]:.2f})")

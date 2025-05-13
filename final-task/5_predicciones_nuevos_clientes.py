import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# cargar datos y preparar el modelo
df = pd.read_csv('datos_prestamos.csv')
X = df.drop('Aprobado', axis=1)
y = df['Aprobado']

# entrenar el modelo (el mismo usado anteriormente)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
modelo = LogisticRegression()
modelo.fit(X_train, y_train)

# nuevos clientes para hacer predicciones
nuevos_clientes = pd.DataFrame({
    'Edad': [30, 45, 22, 40, 52, 33],
    'Ingresos': [3200, 7500, 2100, 3000, 100000, 50000],
    'Puntuacion_Credito': [670, 800, 590, 650, 750, 780],
    'Prestamos_Previos': [1, 2, 0, 0, 3, 1]
})

# hacer predicciones
predicciones = modelo.predict(nuevos_clientes)

# mostrar resultados
nuevos_clientes['Prediccion_Aprobado'] = predicciones
print(nuevos_clientes)

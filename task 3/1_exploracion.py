import pandas as pd

# cargar el archivo CSV
df = pd.read_csv('datos_prestamos.csv')

# Mostrar las primeras filas del DataFrame
print("Primeras filas del dataset:")
print(df.head())

# Ver información general del dataset
print("\nInformación general del dataset:")
print(df.info())

# Estadísticas descriptivas
print("\nEstadísticas descriptivas:")
print(df.describe())

# Verificar si hay valores nulos
print("\nValores nulos por columna:")
print(df.isnull().sum())

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cargar los datos desde el CSV
df = pd.read_csv('datos_prestamos.csv')

# Separar características (x) y variable objetivo (y)
X = df.drop('Aprobado', axis=1)
y = df['Aprobado']

# Dividir en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizar los datos con el standardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Verificación
print("Tamaño de X_train_scaled:", X_train_scaled.shape)
print("Tamaño de X_test_scaled:", X_test_scaled.shape)

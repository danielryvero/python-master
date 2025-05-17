import pandas as pd
from sklearn.model_selection import train_test_split

# cargamos los datos
df = pd.read_csv('datos_prestamos.csv')

# separa caracteristicas (x) y variable objetivo (y)
X = df.drop('Aprobado', axis=1)
y = df['Aprobado']

# dividir en conjunto de entrenamiento y de prueba (70% a 30% para este caso)
X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size=0.3, random_state=42)

# mostrar formas para confirmar que esta correcto
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


print ("""Las shapes son de (35,4) y (15,4) para entrenamiento y test respectivamente, 
como era de esperar debido a la division de conjuntos de prueba y entrenamiento que realizamos""")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# cargar datos despues de ojearlos y notar separador
df = pd.read_csv("Estudiodecaso6.csv", sep=';')

# comprobar valores nulos en el dataset (si da diferente a 0 la suma es que hay nulos)
print("\nValores nulos por columna:")
print(df.isnull().sum())

# histograma
df.hist(figsize=(8, 6))
plt.tight_layout()
plt.show()

# mapa de calor de correlación
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Mapa de calor de correlaciones")
plt.show()

# ---------------------------
# 1. CARGA Y EXPLORACIÓN DE DATOS
# ---------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Estudiodecaso6.csv", sep=';')

print("\nValores nulos por columna:")
print(df.isnull().sum())

# Histograma
df.hist(figsize=(8, 6))
plt.tight_layout()
plt.show()

# Mapa de calor de correlación
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Mapa de calor de correlaciones")
plt.show()

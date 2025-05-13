import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


#############################
# análisis e interpretación #
#############################

print("\nInterpretaciones:")
print(f"- Edad promedio del cliente: {df['Edad'].mean():.2f} años.")
print(f"- Ingreso mensual promedio: {df['Ingresos'].mean():,.2f}.")
print(f"- Puntuación de crédito media: {df['Puntuacion_Credito'].mean():.2f}.")
print(f"- Promedio de préstamos previos: {df['Prestamos_Previos'].mean():.2f}.")
# estas interpretaciones nos dan una idea de las caracteristicas de  los datos 
# y las proporciones de aprobados y no aprobados


aprobados = df['Aprobado'].value_counts(normalize=True)
print(f"- Porcentaje de clientes aprobados: {aprobados.get(1,0)*100:.2f}%.")
print(f"- Porcentaje de clientes no aprobados: {aprobados.get(0,0)*100:.2f}%.")



# boxplots para identificar outliers por clase
plt.figure(figsize=(12, 8))
for i, col in enumerate(['Edad', 'Ingresos', 'Puntuacion_Credito', 'Prestamos_Previos']):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x='Aprobado', y=col, data=df)
    plt.title(f'{col} vs Aprobación')
plt.tight_layout()
plt.show()
# estos boxplots nos permiten ver visualmente qué tanto importan las 
# variables en función de la toma de la decisión


# correlación para mostrar en un heatmap el de las variables para tomar la decisión
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de Correlación")
plt.show()


# reflexion:
print("""
Si la puntuación de crédito o ingresos presentan fuerte correlación con 'Aprobado', 
pueden ser factores decisivos.
Si hay muchas observaciones extremas (outliers), podrían distorsionar el modelo""")

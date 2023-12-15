# Importar las bibliotecas necesarias
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Generar datos de prueba: notas de alumnos
# 200 notas generadas aleatoriamente con una distribución normal
# Media = 70, Desviación estándar = 10
notas = np.random.normal(70, 10, 100)

# Crear el diagrama de cajas con Seaborn
sns.boxplot(x=notas)

# Personalizar el gráfico
plt.title("Distribución de Notas de Alumnos")  # Título del gráfico
plt.xlabel("Notas")  # Etiqueta del eje X

# Mostrar el gráfico
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Generando datos aleatorios con distribución normal
np.random.seed(0)  # Para resultados reproducibles
datos_grupo1 = np.random.normal(100, 10, 1000)  # Media 100, Desviación estándar 10, 1000 datos
datos_grupo2 = np.random.normal(90, 20, 1000)   # Media 90, Desviación estándar 20, 1000 datos

# Creando el histograma
bins = np.linspace(50, 150, 30)  # Define los límites de las barras

plt.hist(datos_grupo1, bins, alpha=0.5, label='Grupo 1')
plt.hist(datos_grupo2, bins, alpha=0.5, label='Grupo 2')

# Añadiendo títulos y etiquetas
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.title('Comparación de Distribuciones')
plt.legend()

# Añadiendo una cuadrícula
plt.grid(True)

# Mostrando el histograma
plt.show()

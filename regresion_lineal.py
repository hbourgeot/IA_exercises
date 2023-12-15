import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Datos de ejemplo (deberías normalizar tus datos reales)
tamanos_casa = np.array([50, 60, 70, 80, 90, 100, 400, 533, 1290, 190], dtype=float)
precios_casa = np.array([30000, 36000, 39000, 42000, 48000, 45000, 54000, 58000, 80000, 30000], dtype=float)

# Normalizar los datos
tamanos_casa_norm = (tamanos_casa - tamanos_casa.mean()) / tamanos_casa.std()
precios_casa_norm = (precios_casa - precios_casa.mean()) / precios_casa.std()

# Crear el modelo
modelo = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# Compilar el modelo con un learning rate más bajo
optimizer = keras.optimizers.SGD(learning_rate=0.01)
modelo.compile(optimizer=optimizer, loss='mean_squared_error')

# Entrenar el modelo
historial = modelo.fit(tamanos_casa_norm, precios_casa_norm, epochs=500, verbose=0)

# Gráfico para ver el proceso de entrenamiento
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(historial.history['loss'])

# Usar el modelo para hacer predicciones normalizadas
tamanos_para_predecir = np.array([120, 150])  # Por ejemplo, casas de 120m² y 150m²
tamanos_para_predecir_norm = (tamanos_para_predecir - tamanos_casa.mean()) / tamanos_casa.std()
precios_predichos_norm = modelo.predict(tamanos_para_predecir_norm)
# Desnormalizar las predicciones
precios_predichos = precios_predichos_norm * precios_casa.std() + precios_casa.mean()
print("Predicciones:", precios_predichos)

# Mostrar el gráfico
plt.show()

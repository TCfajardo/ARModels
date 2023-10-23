import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm # para análisis de series temporales.

# Generar un proceso AR(1)
np.random.seed(0)
n = 500         # es la longitud de la serie temporal que queremos generar.
phi = 0.9       # es el coeficiente autorregresivo que define el proceso AR(1).
w = np.random.normal(0, 1, n)   # es un vector de ruido blanco (valores aleatorios con distribución normal estándar).
x = np.zeros(n)     # es un vector lleno de ceros que contendrá la serie temporal generada.

#calcula los valores de la serie temporal AR(1) utilizando la ecuación del modelo AR(1). 
# Comienza desde t=1 ya que necesitamos un valor inicial para comenzar la recursión.
for t in range(1, n):
    x[t] = phi * x[t - 1] + w[t]

# Visualizar el proceso
plt.figure(figsize=(8, 4))
plt.plot(x, label="Proceso AR(1)")
plt.xlabel("Tiempo")
plt.ylabel("Valor")
plt.title("Proceso Autorregresivo AR(1)")
plt.legend()
plt.show()

# Análisis de autocorrelación
acf = sm.tsa.acf(x, nlags=40)
plt.figure(figsize=(8, 4))
plt.stem(acf, use_line_collection=True)
plt.xlabel("Lag")
plt.ylabel("Autocorrelación")
plt.title("Autocorrelación del proceso AR(1)")
plt.show()

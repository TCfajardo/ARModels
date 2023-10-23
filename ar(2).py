import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Generar un proceso AR(2)
np.random.seed(0)
n = 500
phi1, phi2 = 0.6, -0.3  # Coeficientes autorregresivos
w = np.random.normal(0, 1, n)
x = np.zeros(n)

# Calcula los valores de la serie temporal AR(2) utilizando la ecuación del modelo AR(2).
# Comienza desde t=2 ya que necesitamos dos valores iniciales para comenzar la recursión.
x[0] = w[0]
x[1] = w[1]

for t in range(2, n):
    x[t] = phi1 * x[t - 1] + phi2 * x[t - 2] + w[t]

# Visualizar el proceso
plt.figure(figsize=(8, 4))
plt.plot(x, label="Proceso AR(2)")
plt.xlabel("Tiempo")
plt.ylabel("Valor")
plt.title("Proceso Autorregresivo AR(2)")
plt.legend()
plt.show()

# Análisis de autocorrelación
acf = sm.tsa.acf(x, nlags=40)
plt.figure(figsize=(8, 4))
plt.stem(acf, use_line_collection=True)
plt.xlabel("Lag")
plt.ylabel("Autocorrelación")
plt.title("Autocorrelación del proceso AR(2)")
plt.show()

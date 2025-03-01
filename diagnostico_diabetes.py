#Descripcion neuronal artificial simple donde calculo diagnosticos de pacientes de su glucosa para predecir si son propensos a contraer diabetes.
#Autor: Dr. Aldo Gonzalez Vazquez
#Version: 0.1
#Fecha 01/03/2025
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Crear un dataset simple (simulación de datos médicos)
# Características: [edad, nivel de glucosa en sangre]
X = np.array([
    [25, 80], [30, 95], [45, 120], [50, 150], 
    [60, 180], [65, 140], [70, 160], [75, 200]
])
# Etiquetas: 0 = No tiene diabetes, 1 = Tiene diabetes
y = np.array([0, 0, 0, 1, 1, 0, 1, 1])

# 2. Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 3. Normalizar los datos (escalar para mejorar el rendimiento del modelo)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Crear el modelo de red neuronal
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))  # Capa oculta con 4 neuronas
model.add(Dense(1, activation='sigmoid'))            # Capa de salida (clasificación binaria)

# 5. Compilar el modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 6. Entrenar el modelo
model.fit(X_train, y_train, epochs=100, verbose=0)

# 7. Evaluar el modelo en el conjunto de prueba
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Pérdida en el conjunto de prueba: {loss}")
print(f"Precisión en el conjunto de prueba: {accuracy}")

# 8. Hacer predicciones
nuevos_datos = np.array([[40, 110], [55, 170]])  # Nuevos datos para predecir
nuevos_datos = scaler.transform(nuevos_datos)    # Normalizar los nuevos datos
predicciones = model.predict(nuevos_datos)
print("Predicciones para nuevos datos:")
print(predicciones)
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 13:32:21 2024

@author: Fantasma
"""

import tensorflow as tf
import os
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import cv2
import numpy as np
#  ruta de  dataset etiquetado 
directorio_dataset = "C:/Users/Fantasma/Downloads/rostro"


#leer etiquetado
ruta = os.path.join(directorio_dataset, "etiquetas.csv")
df_eti = pd.read_csv(ruta)

# configuracion de la red neuronal
modelo = models.Sequential()
modelo.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(160, 160, 3)))
modelo.add(layers.MaxPooling2D((2, 2)))
modelo.add(layers.Conv2D(64, (3, 3), activation='relu'))
modelo.add(layers.MaxPooling2D((2, 2)))
modelo.add(layers.Conv2D(128, (3, 3), activation='relu'))
modelo.add(layers.MaxPooling2D((2, 2)))
modelo.add(layers.Flatten())
modelo.add(layers.Dense(128, activation='relu'))
modelo.add(layers.Dense(11, activation='softmax')) #asumiendo 11 clases 
#modelo.add(layers.Dense(len(os.listdir(directorio_dataset)), activation='softmax'))  # Asumiendo una clase por persona

modelo.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])




# Configuracion del generador de datos para la carga de imágenes
generador_entrenamiento = ImageDataGenerator(rescale=1./255, validation_split=0.2)

tamaño_lote = 2

df_eti['etiquetas'] = df_eti[['bbox_x', 'bbox_y', 'bbox_width', 'bbox_height']].astype(str).agg(','.join, axis=1)

conjunto_entrenamiento = generador_entrenamiento.flow_from_dataframe(
    df_eti,
    directory=directorio_dataset,
    x_col="image_name",
    y_col="etiquetas",
    target_size=(160, 160),
    batch_size=tamaño_lote,
    class_mode='categorical',
    subset='training'
)

# if conjunto_entrenamiento.samples < tamaño_lote:
#     print("No hay suficientes muestras para el tamaño del lote. Aumenta el tamaño del conjunto de datos o reduce el tamaño del lote.")
# else:
#     # Entrenar el modelo
#     historial_entrenamiento = modelo.fit(
#         conjunto_entrenamiento,
#         steps_per_epoch=conjunto_entrenamiento.samples // tamaño_lote,
#         epochs=10,  # Ajusta según sea necesario
#     )
    
# train del modelo
total_muestras = conjunto_entrenamiento.samples

historial_entrenamiento = modelo.fit(
    conjunto_entrenamiento,
    steps_per_epoch=conjunto_entrenamiento.samples // tamaño_lote,
    epochs=10,  # Ajustar según sea necesario
)


 #pruebas
 
imagen = "C:/Users/Fantasma/Downloads/img1.jpeg"
 
imagen = cv2.imread("C:/Users/Fantasma/Downloads/img1.jpeg")
imagen = cv2.resize(imagen, (160, 160))  # Ajustar al tamaño de entrada del modelo
imagen = imagen / 255.0  # Normalizar los valores de píxeles

# Expandir las dimensiones para que coincidan con las expectativas del modelo
imagen = np.expand_dims(imagen, axis=0)

# Realizar la predicción
predicciones = modelo.predict(imagen)

# Imprimir las predicciones
print("Predicciones:", predicciones)
 

clase_predicha = np.argmax(predicciones)
print("Clase Predicha:", clase_predicha)

# Guardar el modelo entrenado
modelo.save("modelo_reconocimiento_facial.h5")

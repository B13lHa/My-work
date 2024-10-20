import tensorflow as tf
from keras import layers, models
import os
import numpy as np
import cv2
import random

width = 100
height = 100

ruta_train = 'malaria-cells/train/'
ruta_predict = 'malaria-cells/test/29.png'

train_x = []
train_y = []

labels = os.listdir(ruta_train)
'''
for i in os.listdir(ruta_train):
    for j in os.listdir(ruta_train + i):
        img = cv2.imread(ruta_train+i+'/'+j)
        resized_image = cv2.resize(img, (100, 100))

        train_x.append(resized_image)

        for x,y in enumerate(labels):
            if y == i:
                array = np.zeros(len(labels))
                array[x]=1
                train_y.append(array)

x_data = np.array(train_x)
y_data = np.array(train_y)

model = tf.keras.Sequential([
    layers.Conv2D(32, 3, 3, input_shape=(100,100,3), data_format='channels_first'),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2,2), padding='same'),
    layers.Conv2D(32,3,3, padding='same'),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Conv2D(64,3,3, padding='same'),
    layers.Activation('relu'), 
    layers.MaxPooling2D(pool_size=(2,2), padding='same'),
    layers.Flatten(),
    layers.Dense(64),
    layers.Activation('relu'),
    layers.Dropout(0.5),
    layers.Dense(2),
    layers.Activation('sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

epochs = 100

model.fit(x_data, y_data, epochs = epochs)

models.save_model(model, 'mimodelo.keras')
'''
model = models.load_model('mimodelo.keras')
my_image = cv2.imread(ruta_predict)
my_image = cv2.resize(my_image, (100, 100))

resultat = model.predict(np.array([my_image]))[0]

porcentaje = max(resultat)*100

grupo = ''

if resultat.argmax()==0:
    grupo = 'Unparasitized'
else:
    grupo = 'Parasitized'

print("Puc dir amb un " + porcentaje + "% que aquesta cèl·lula és" + grupo)



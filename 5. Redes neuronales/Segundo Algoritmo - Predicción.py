import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

#Definimos la misma altura que definimos en el entrenamiento
longitud, altura = 150, 150
modelo = './modelo/modelo.h5'
pesos_modelo = './modelo/pesos.h5'

#Usaremos el modelo que generamos en el entrenamiento
cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)

#Caragaremos la imagena a predecir
def predict(file):
    x = load_img(file, target_size=(longitud, altura))
    x = img_to_array(x) #Convertimos a un arreglo nuestra imagen
    x = np.expand_dims(x, axis=0) #Anadmios una dimension estra para
    un mejor procesamiento
    array = cnn.predict(x) # llamamos a la red neural convolucional
    result = array[0] # toamos solo la primera dimension del arreglo
    answer = np.argmax(result) # indice del valor mas alto en el
    resultado
    if answer == 0:
        print("pred: Perro")
    elif answer == 1:
        print("pred: Gato")
    elif answer == 2:
        print("pred: Gorila")
    return answer
predict('cat.jpg')

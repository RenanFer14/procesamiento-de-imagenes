import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator # preprocesamiento de imagenes
from tensorflow.python.keras.optimizers import Adam #Optmizador del modelo
from tensorflow.python.keras.models import Sequential # libreria quenos permite hacer redes neuronales secuenciales
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D # Capas para las convoluciones y polling
from tensorflow.python.keras import backend as K

# Cerramos cualquier secion abierta de keras
K.clear_session()
data_entrenamiento = './data/entrenamiento'
data_validacion = './data/validacion'

#parametros
epocas=20 #nuemro de veces de interacion sobre todo el set de datos
longitud, altura = 150, 150 #tamano en el cual se le da la imagen(pixeles)
batch_size = 32 #numero de imagenes que se enviaran a procesar en cada paso
pasos = 250 # numero de veces que se procesara la informacion en cada epoca
validation_steps = 50 # 200 pasos con el de datos de validacion
filtrosConv1 = 32 #despues de la rpimera convolucion este tendra 32 de profundidad
filtrosConv2 = 64
tamano_filtro1 = (3, 3) # Tamano del filtro
tamano_filtro2 = (2, 2)
tamano_pool = (2, 2) # tamano del fltro del maxpooling
clases = 3 # numero de clases(gato, perro, gorila)
lr = 0.0004 # Ajuste para una solucion optima

#Preparamos nuestras imagenes
entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255, # pasamos la escala a 0 y 1
    shear_range=0.2, # inclinacion de imagenes
    zoom_range=0.2, # realizamos un zoon en algunas imagenes
    horizontal_flip=True) # invierte la imagen

test_datagen = ImageDataGenerator(rescale=1. / 255) # en la validacion solo reesccalamos

entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento, # nos dirigimos a la carpeta
    target_size=(altura, longitud), # todas las procesamos a una altura y longitus especifica
    batch_size=batch_size,
    class_mode='categorical') # La clasificacino sera categorica solo con tres etiquetas

validacion_generador = test_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')

# Creamos la CNN
cnn = Sequential() # La red neuuronal es secuencial
# Primera capa es una convolucion
cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding ="same",input_shape=(longitud, altura, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))
# Segunda capa convolucional
cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding="same")) #activation='relu')
cnn.add(MaxPooling2D(pool_size=tamano_pool))

#Clasificacion
cnn.add(Flatten()) #Pasamos la imagen de varias profundiadas a una sola (aplanamos la imagen)
cnn.add(Dense(256, activation='relu')) # la anadimos a una ultima capa con 256 neuronas
cnn.add(Dropout(0.5)) #Apagamos el 50% de neuronas, para evitar un sobreajuste
cnn.add(Dense(clases, activation='softmax')) # la andimos a una ultima capa con el # de neruonas igaula al de las clases
# Mejoramos el porcentaje de imagens bien clasificadas
cnn.compile(loss='categorical_crossentropy',
            optimizer = Adam(lr=lr),
            metrics=['accuracy'])
#Entramos cada paso durante cada epoca, para genrrar el modelo
cnn.fit_generator(
    entrenamiento_generador,
    steps_per_epoch=pasos,
    epochs=epocas,
    validation_data=validacion_generador,
    validation_steps=validation_steps)

# Guardamos nuestro modelo
target_dir = './modelo/'
if not os.path.exists(target_dir): # Creamos el directorio
    os.mkdir(target_dir)
cnn.save('./modelo/modelo.h5') #Estructura del modelo
cnn.save_weights('./modelo/pesos.h5') #Peso en cada una de las capas que entrenamos



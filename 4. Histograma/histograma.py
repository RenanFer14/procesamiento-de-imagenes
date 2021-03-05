#·······························································#
#·   UNIVERSIDAD NACIONAL SAN ANTONIO ABAD DEL CUSCO            #
#·   Escuela Profesional de Ingenieria Informatica y de Sistemas#
#·   Robotica y Procesamiento de Señales                        #
#·   Reconocimiento de objetos por medio del histograma o filtro#
#    segun los colores del objeto                               #
#·······························································#


# Libreiras necesarias
import cv2
import numpy as np
import urllib
import urllib.request


# definimos un procedimiento para dibujar contornos y hallar coordenadas
def dibujar(mask, color):
    contornos, nul = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Se recorre cada uno de los contornos encontrados
    for c in contornos:
        # Determinamos el area del contorno en pixels 
        area = cv2.contourArea(c)
        # Si el area es mayor al valor se dibuja
        if area > 500:
            # buscamos areas centrales
            M = cv2.moments(c)
            # un if por si el denominador es 0 y pueda ser indeterminado
            if (M["m00"] == 0): 
                M["m00"] = 1
            # Encontramos coordenadas X y Y
            x = int(M["m10"] / M["m00"])
            y = int(M['m01'] / M['m00'])
            # mejora la visualizacion del contorno
            nuevoContorno = cv2.convexHull(c)
            # dibujamos un circulo de 7 pixeles en el centro del objeto encontrado XY 
            cv2.circle(image, (x, y), 7, (0, 255, 0), -1)
            # para visualizar el texto
            cv2.putText(image, '{},{}'.format(x, y), (x + 10, y), font, 0.75, (0, 255, 0), 1, cv2.LINE_AA)
            # dibujamos los contornos
            cv2.drawContours(image, [nuevoContorno], 0, color, 3)


# Creamos una variable de conexion IP con la camara
url = 'http://192.168.1.153:8080/shot.jpg'

# Definimos el rango de valores del azul
low_blue = np.array([100, 100, 20], np.uint8)
high_blue = np.array([125, 255, 255], np.uint8)

# Definimos el rango de valores del amarillo
low_yellow = np.array([20, 100, 20], np.uint8)
high_yellow = np.array([45, 255, 255], np.uint8)

# Definimos el rango de valores del rojo
low_red_1 = np.array([0, 100, 20], np.uint8)
high_red_1 = np.array([5, 255, 255], np.uint8)

low_red_2 = np.array([175, 100, 20], np.uint8)
high_red_2 = np.array([179, 255, 255], np.uint8)

# Definimos el rango de valores del Naranja
low_orange = np.array([11, 100, 20], np.uint8)
high_orange = np.array([19, 255, 255], np.uint8)

# Definimos el rango de valores del Verde
low_green = np.array([36, 100, 20], np.uint8)
high_green = np.array([70, 255, 255], np.uint8)

# Definimos el rango de valores del Violeta
low_violet = np.array([130, 100, 20], np.uint8)
high_violet = np.array([145, 255, 255], np.uint8)

# Definimos el rango de valores del Rosa
low_pink = np.array([146, 100, 20], np.uint8)
high_pink = np.array([170, 255, 255], np.uint8)

font = cv2.FONT_HERSHEY_SIMPLEX

# Iniciamos el programa en un bucle
while True:
    # Iniciamos la transmision de la camara
    imgResp = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)

    # Almacenamos la transmision en una variable image
    image = cv2.imdecode(imgNp, -1)

    # Transformamos la imagen de BGR a HSV
    frameHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #OBTENCION DE MASCARAS DE COLOR
    # Obtenemos una imagen binaria del azul
    maskAzul = cv2.inRange(frameHSV, low_blue, high_blue)

    # Obtenemos una imagen binaria del amarillo
    maskAmarillo = cv2.inRange(frameHSV, low_yellow, high_yellow)

    # Obtenemos una imagen binaria del rojo 1
    maskRed1 = cv2.inRange(frameHSV, low_red_1, high_red_1)

    # Obtenemos una imagen binaria del rojo 2
    maskRed2 = cv2.inRange(frameHSV, low_red_2, high_red_2)

    # Obtenemos una imagen binaria del rojo total
    maskRed = cv2.add(maskRed1, maskRed2)

    # Obtenemos una imagen binaria del Naranja
    maskNaranja = cv2.inRange(frameHSV, low_orange, high_orange)

    # Obtenemos una imagen binaria del verde
    maskVerde = cv2.inRange(frameHSV, low_green, high_green)

    # Obtenemos una imagen binaria del violeta
    maskVioleta = cv2.inRange(frameHSV, low_violet, high_violet)

    # Obtenemos una imagen binaria del rosa
    maskRosa = cv2.inRange(frameHSV, low_pink, high_pink)

    # Dibujamos los contornos encontrados segun su color
    dibujar(maskAzul, (255, 0, 0))
    dibujar(maskAmarillo, (0, 255, 255))
    dibujar(maskRed, (0, 0, 255))
    dibujar(maskNaranja, (26, 127, 239))
    dibujar(maskVerde, (0, 255, 0))
    dibujar(maskVioleta, (120, 40, 140))
    dibujar(maskRosa, (255, 0, 255))
    
    # mostramos el video
    cv2.imshow('frame', image)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

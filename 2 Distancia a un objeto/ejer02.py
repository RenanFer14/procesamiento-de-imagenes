#Distancia entre el objeto y la camara
# Importamos las debidas librerias 
import cv2
import numpy as np
import urllib.request

def Marcador_imagen(imagen):
    # usamos un desenfoque gauseano para obtener los bordes
    desenfoque = cv2.GaussianBlur(imagen, (5, 5), 0)
    #canbiamos el color BGR a HSV
    color_hsv = cv2.cvtColor(desenfoque, cv2.COLOR_BGR2HSV)
    # definimos el rango del color que queremos que detecte
    lower_blue = np.array([38, 86, 0])
    upper_blue = np.array([121, 255, 255])

    mask = cv2.inRange(color_hsv, lower_blue, upper_blue)
    #definimos el contorno de la imagen
    contorno, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    con = max(contorno, key=cv2.contourArea)

    return cv2.minAreaRect(con)

#definimos la distancia 
def Distancia_Camara(anchoConocido, longitudFocal, distanciaPorConocer):
    # calculamos la distancia de la camara al objeto 
    return (anchoConocido * longitudFocal) / distanciaPorConocer

# definimos el ancho y la distancia para hallas la longitud focal 
DistanciaConocida = 20.0
anchoConocido = 3.0

# ponemos la url de nuestro servidor 

url = 'http://192.168.1.4:8080/shot.jpg'

#tranformamos la imagen en bytearray
imgResp = urllib.request.urlopen(url)
imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
imagen1 = cv2.imdecode(imgNp, -1)
#hallamos el marcador de la imagen usando la longitud focal
marker = Marcador_imagen(imagen1)
longitudFocal = (marker[1][0] * DistanciaConocida) / anchoConocido

while True:
    #capturamos nuestra imagen
    imgResp = urllib.request.urlopen(url)
    imagen = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    imagen = cv2.imdecode(imagen, -1)
    #tranformamos a escala de grises
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    marker = Marcador_imagen(imagen)
    CM = Distancia_Camara(anchoConocido, longitudFocal, marker[1][0])
    # escribimos la salida
    cv2.putText(imagen, "%.2fcm" % CM,
                (imagen.shape[1] - 350, imagen.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (225, 0, 0), 3)
    #imagen en el frame
    cv2.imshow("imagen", imagen)
    key = cv2.waitKey(1)
    # finalizamos el programa 
    if key == 27:
        break
cv2.destroyAllWindows()

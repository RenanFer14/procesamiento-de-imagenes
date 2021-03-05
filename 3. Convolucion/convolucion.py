#·······························································#
#·   UNIVERSIDAD NACIONAL SAN ANTONIO ABAD DEL CUSCO            #
#·   Escuela Profesional de Ingenieria Informatica y de Sistemas#
#·   Robotica y Procesamiento de Señales                        #
#·   Algoritmo de convolucion para aplicar sobre imagenes       #
#·······························································#

#Librerias necesarias
import cv2
import numpy as np
import urllib
import urllib.request
import urlopen

url = 'http://192.168.1.153:8080/shot.jpg'
cap = cv2.VideoCapture(1)
while (1):
    lower_red = np.array([30, 150, 50])
    upper_red = np.array([255, 255, 180])

    imgResponse = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgResponse.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)
    # ---------------------------------------------
    mask = cv2.inRange(img, lower_red, upper_red)
    res = cv2.bitwise_and(img, img, mask=mask)
    # ---------------------------------------------

    #Filtros
    #Laplaciano
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    #Filtro Sobel X
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    #Filtro Sobel Y
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    #Filtro de bordes
    edges = cv2.Canny(img, 100, 200)

    #Mostrar ventanas de imagenes
    cv2.imshow('Original', img)
    cv2.imshow('Mask', mask)
    cv2.imshow('Filtro Laplaciano', laplacian)
    cv2.imshow('Filtro Sobel X', sobelx)
    cv2.imshow('Filtro Sobel Y', sobely)
    cv2.imshow('Filtro de bordes - Edges', edges)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()

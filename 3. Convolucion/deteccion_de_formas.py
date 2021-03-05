#·······························································#
#·   UNIVERSIDAD NACIONAL SAN ANTONIO ABAD DEL CUSCO            #
#·   Escuela Profesional de Ingenieria Informatica y de Sistemas#
#·   Robotica y Procesamiento de Señales                        #
#·   Algoritmo de convolucion para aplicar sobre imagenes       #
#·   Deteccion de figuras                                       #
#·······························································#

#Librerias necesarias
import cv2
import numpy as np
import urllib
import urllib.request
import urlopen


def pasar(x):
    pass


url = 'http://192.168.1.153:8080/shot.jpg'

# Instrucciones para aumentar la intensidad
cv2.namedWindow("TrackBars")
cv2.createTrackbar("L-H", "TrackBars", 0, 180, pasar)
cv2.createTrackbar("L-S", "TrackBars", 66, 255, pasar)
cv2.createTrackbar("L-V", "TrackBars", 134, 255, pasar)
cv2.createTrackbar("U-H", "TrackBars", 180, 180, pasar)
cv2.createTrackbar("U-S", "TrackBars", 255, 255, pasar)
cv2.createTrackbar("U-V", "TrackBars", 243, 255, pasar)

# --------------
font = cv2.FONT_HERSHEY_COMPLEX
while True:
    
    l_h = cv2.getTrackbarPos("L-H", "TrackBars")
    l_s = cv2.getTrackbarPos("L-S", "TrackBars")
    l_v = cv2.getTrackbarPos("L-V", "TrackBars")
    u_h = cv2.getTrackbarPos("U-H", "TrackBars")
    u_s = cv2.getTrackbarPos("U-S", "TrackBars")
    u_v = cv2.getTrackbarPos("U-V", "TrackBars")

    lower_red = np.array([l_h, l_s, l_v])
    upper_red = np.array([u_h, u_s, u_v])
    # --------------------------------------------
    imgResponse = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgResponse.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)
    # --------------------------------------------
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_red, upper_red)
    # ----------------------------------------
    # para mejorar las lineas
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)

    # Deteccion de borde
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        # definir area para eliminar ruidos de objetos pequeños en la imagen
        area = cv2.contourArea(cnt)
        # define rectas, cantidad de rectas
        # 0.01: calidad de las rectas que forman un triangulo o cuadrado mas no para circulo circulo
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

        # para definir solo un figuras geo y eliminar otros ruidos
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        if area > 400:  # pixeles
            # ------------
            # define color de bordes(negro)
            cv2.drawContours(img, [approx], 0, (0, 0, 0), 2)

            if len(approx) == 3:
                cv2.putText(img, "Triangulo", (x, y), font, 1, (0, 0, 0))
                print('se detecta triangulo')

            elif len(approx) == 4:
                cv2.putText(img, "Rectangulo", (x, y), font, 1, (0, 0, 0))
                print('se detecta rectangulo')

            elif 10 < len(approx) < 20:
                cv2.putText(img, "Circulo", (x, y), font, 1, (0, 0, 0))
                print('se detecta circulo')

    # -------------------------------------        
    cv2.imshow("Frame", img)
    cv2.imshow("Mask", mask)
    # cv2.imshow("kernel", kernel)
    key = cv2.waitKey(5)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()

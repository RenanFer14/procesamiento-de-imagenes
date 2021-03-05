#Reconocimiento de lineas y circunferencias 
# importamos las librarias 
import cv2
import numpy as np
import urllib.request

#reconocimiento de lineas
# definimos la captura de imagenes
def lineas():
    #ponemos la url para obtener la imagen
    url = 'http://192.168.1.4:8080/shot.jpg'
    while True:
        with urllib.request.urlopen(url) as response:
            entrada = response.read()
        #tranformamos la imagen en bytearray
        imgNp = np.array(bytearray(entrada), dtype=np.uint8)
        img = cv2.imdecode(imgNp, -1)
        cv2.imwrite('opencv.png', img)
        img = cv2.imread('opencv.png')
        #tranformamos a escala de grises
        #plicamos canny para el reconocimientode objetos
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, 3)
        #reconocemos todas las lineas disponobles
        lineas = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        if lineas is None:
            lineas = []
        for linea in lineas:
            #tranformamos a coordenadas cartesianas
            rho, theta = linea[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            #damos los rangos de 1000 a -1000 para poder ver en toda la pantalla
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            #escribimos y mostramos la imagen
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

#Reconocimiento de circulos 
def circulo():
    #ponemos la url para obtener la imagen
    url = 'http://192.168.1.4:8080/shot.jpg'
    while True:
        with urllib.request.urlopen(url) as response:
            entrada = response.read()
        #tranformamos la imagen en bytearray
        imgNp = np.array(bytearray(entrada), dtype=np.uint8)
        img = cv2.imdecode(imgNp, -1)
        cv2.imwrite('opencv.png', img)
        img = cv2.imread('opencv.png')
        #usamos el metodo de  para desenfocar la imagen
        src = cv2.medianBlur(img, 5)
        #tranformamos a escala de grises
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        #HoughCircles para el reconocimiento de circulos
        circles = cv2.HoughCircles(src, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30, minRadius=10, maxRadius=50)
        #si no hay imagen no detiene la iteracion
        if circles is None:
            circles = [[]]
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # dibujar circulo
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # dibujar centro
            cv2.circle(img, (i[0], i[1]), 1, (225, 0, 0), 3)
        cv2.imshow('detected circles', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


circulo()
#lineas()


import os
import imutils
import cv2


nombre = 'Mauro'
ruta = 'C:\\Users\\mauri\\OneDrive\\Escritorio\\Facial\\Data' #Ruta donde se almacena Los datos
CarpetaPersona = ruta + '/' + nombre

if not os.path.exists(CarpetaPersona):
	print('Carpeta creada: ',CarpetaPersona)
	os.makedirs(CarpetaPersona)

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

clasificadorFacial = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
contador = 0

while True:

	ret, ventana = cap.read()
	if ret == False: break
	ventana =  imutils.resize(ventana, width=640)
	gray = cv2.cvtColor(ventana, cv2.COLOR_BGR2GRAY)
	auxventana = ventana.copy()

	faces = clasificadorFacial.detectMultiScale(gray,1.3,5)

	for (x,y,w,h) in faces:
		cv2.rectangle(ventana, (x,y),(x+w,y+h),(0,255,0),2)
		rostro = auxventana[y:y+h,x:x+w]
		rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
		cv2.imwrite(CarpetaPersona + '/rostro_{}.jpg'.format(contador),rostro)
		contador = contador + 1
	cv2.imshow('Capturando Rostro',ventana)

	k =  cv2.waitKey(1)
	if k == 27 or contador >= 300:  
		break

cap.release()
cv2.destroyAllWindows()

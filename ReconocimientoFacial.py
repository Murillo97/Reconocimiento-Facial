import cv2
import os

ruta = 'C:\\Users\\mauri\\OneDrive\\Escritorio\\Facial\\Data' #rua de almacenaciento
carpetasImagen = os.listdir(ruta)
print('carpetasImagen=',carpetasImagen)


face_recognizer = cv2.face.LBPHFaceRecognizer_create()  #Crear el modelo LBPH
face_recognizer.read('modeloLBPHFace.xml')

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

clasificadorFacial = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

while True:
	ret,ventana = cap.read()
	if ret == False: break
	gray = cv2.cvtColor(ventana, cv2.COLOR_BGR2GRAY)
	auxventana = gray.copy()

	cara = clasificadorFacial.detectMultiScale(gray,1.3,5)

	for (x,y,w,h) in cara:
		rostro = auxventana[y:y+h,x:x+w]
		rostro = cv2.resize(rostro,(170,170),interpolation= cv2.INTER_CUBIC) #Tamaño de la imagen 
		resultado = face_recognizer.predict(rostro) #devuelve la etiqueta aprendida para cada objeto 

		# LBPHFace
		if resultado[1] < 70:
			cv2.putText(ventana,'{}'.format(carpetasImagen[resultado[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
			cv2.rectangle(ventana, (x,y),(x+w,y+h),(0,255,0),2) #Rectangulo de la cara con grosor de 2 color verde
			cv2.putText(ventana,'estudiante',(x,y-60),2,0.8,(0,255,0),1,cv2.LINE_AA)
		else:
			cv2.putText(ventana,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
			cv2.rectangle(ventana, (x,y),(x+w,y+h),(0,0,255),2)
		
		
			
	cv2.imshow('Reconocimiento',ventana)
	k = cv2.waitKey(1)
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()
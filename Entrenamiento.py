import cv2
import os
import numpy as np

ruta = 'C:\\Users\\mauri\\OneDrive\\Escritorio\\Facial\\Data' #Ruta almacenamiendo
listadoPersonas = os.listdir(ruta)
print('Lista de personas: ', listadoPersonas)

etiquetaPersona = []
Datos = []
ContadorRostro = 0

for Directorio in listadoPersonas:
	CarpetaPersona = ruta + '/' + Directorio
	print('Analizando..')

	for NombreArchivo in os.listdir(CarpetaPersona):
		print('Rostros: ', Directorio + '/' + NombreArchivo)
		etiquetaPersona.append(ContadorRostro)
		Datos.append(cv2.imread(CarpetaPersona+'/'+NombreArchivo,0))
		
	ContadorRostro = ContadorRostro + 1


face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Entreno del Reconocedor
print("El reconocedor facial se Entreno")
face_recognizer.train(Datos, np.array(etiquetaPersona))

# Modelo Almacenado
face_recognizer.write('modeloLBPHFace.xml')

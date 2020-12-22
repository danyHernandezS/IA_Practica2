# load and display an image with Matplotlib
#from matplotlib import image
#from matplotlib import pyplot
#from os import listdir

from PIL import Image
import numpy as np
import glob
from numpy import full

#Obtener la data de USAC
#path: la url de la carpeta a cargar
#y_val: valor de salida a concatenar al final de cada coleccion de datos
#valores de y_val: 1=correcto, 0=incorrecto
def getArrayByPath(path, y_val):

    filelist = glob.glob(path+"*.jpg")

    pdata = []
    #leeo los archivos
    for filename in filelist:
        image = Image.open(filename)
        data = np.asarray(image)    #convierte archivos a (128 x 128 x 3)
        pdata.append(data)

    npdata = np.array(pdata)
    test_data = npdata.reshape(npdata.shape[0], -1) # lo aplano
    salida = np.full((test_data.shape[0],test_data.shape[1] + 1),y_val) 
    salida[:, :-1] = test_data #agregarle la salida
    return salida


def load_dataset_Usac():
    data_set_usac = getArrayByPath('Imagenes/USAC/',1)
    data_set_mariano = getArrayByPath('Imagenes/Mariano/',0)
    data_set_marro = getArrayByPath('Imagenes/Marroquin/',0)
    data_set_landivar = getArrayByPath('Imagenes/Landivar/',0)

    data = np.concatenate((data_set_usac, data_set_mariano, data_set_marro, data_set_landivar), axis=0)
    np.random.shuffle(data)

    separador = int(len(data) * 0.8) #Obtenemos el valor del 80% de los datos para hacer el slice

    train_data_x = data[:separador,:49152]
    train_data_y = data[:separador,49152:]
    #test_data = data[separador:]


    print(data.shape)
    print(train_data_x.shape)
    print(train_data_y.shape)

    return data

def load_dataset_Mariano():
    data_set_usac = getArrayByPath('Imagenes/USAC/',0)
    data_set_mariano = getArrayByPath('Imagenes/Mariano/',1)
    data_set_marro = getArrayByPath('Imagenes/Marroquin/',0)
    data_set_landivar = getArrayByPath('Imagenes/Landivar/',0)

    print(data_set_usac.shape)
    print(data_set_mariano.shape)
    print(data_set_marro.shape)
    print(data_set_landivar.shape)
    
    data = np.concatenate((data_set_usac, data_set_mariano, data_set_marro, data_set_landivar), axis=0)
    np.random.shuffle(data)
    return data

def load_dataset_Marro():
    data_set_usac = getArrayByPath('Imagenes/USAC/',0)
    data_set_mariano = getArrayByPath('Imagenes/Mariano/',0)
    data_set_marro = getArrayByPath('Imagenes/Marroquin/',1)
    data_set_landivar = getArrayByPath('Imagenes/Landivar/',0)

    print(data_set_usac.shape)
    print(data_set_mariano.shape)
    print(data_set_marro.shape)
    print(data_set_landivar.shape)
    
def load_dataset_Landivar():
    data_set_usac = getArrayByPath('Imagenes/USAC/',0)
    data_set_mariano = getArrayByPath('Imagenes/Mariano/',0)
    data_set_marro = getArrayByPath('Imagenes/Marroquin/',0)
    data_set_landivar = getArrayByPath('Imagenes/Landivar/',1)

    print(data_set_usac.shape)
    print(data_set_mariano.shape)
    print(data_set_marro.shape)
    print(data_set_landivar.shape)

    data = np.concatenate((data_set_usac, data_set_mariano, data_set_marro, data_set_landivar), axis=0)
    np.random.shuffle(data)
    return data


load_dataset_Usac()


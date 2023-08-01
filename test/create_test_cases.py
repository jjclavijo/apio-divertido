import toml
import os

import eletor.parameter_generation as pgen
from call_old_hector import call_hector_with_toml
from tempfile import TemporaryDirectory

import numpy as np

# Esto debería estar en algun constants o algo así, pero bueno. En C sería un
# enum.

MODELS = ['White','Powerlaw','Flicker','RandomWalk','AR1','VaryingAnnual','GGM','Matern']

def create_model_data(n,model):
    """
    Un generador que devuelve hasta n juegos de parámetros para el modelo de
    ruido indicado. No importa el case del nombre del modelo (pero va sin
    guiones bajos).
    """
    model = model.lower()

    # Buscar la función del modelo
    model_fun = getattr(pgen,model)

    for i in range(n):
        # Se cambia la etiqueta con que cada modelo se genera para que en el
        # toml no se pisen las etiquetas.
        label = f'{i}'
        yield label,model_fun(label,m=20)


def main(n):
    """
    Itera sobre todos los modelos y crea n ejemplos de cada uno con su MD5 de
    la salida de hector incluido.
    """
    data_to_write = []
    # Vamos a trabajar con directorios temporales, asique guardamos el actual.
    cwdir = os.getcwd()

    # Por cada modelo
    for model in MODELS:
        # para cada uno de 10 juegos de datos.
        for l,data in create_model_data(n,model):
            # Escupimos basura a la pantalla.
            # print(l,data)

            # Agregamos las opciones generales de configuración que hector
            # Si o Si necesita.

            # Redondeamos el intervalo de tiempo para evitar
            # errores de redondeo que después molesten a los test_cases.
            dt = data['NoiseModels'][model][l]['dt']
            dt = float(np.round(dt,5)) # a toml no le gustan numpy.float64
            data['NoiseModels'][model][l]['dt'] = dt

            data['general'] = {}
            data['file_config'] = {}
            data['general']['NumberOfSimulations'] = 1
            data['general']['TimeNoiseStart'] = 0
            data['general']['NumberOfPoints'] = data['NoiseModels'][model][l]['m']
            data['general']['SamplingPeriod'] = data['NoiseModels'][model][l]['dt']
            data['file_config']['SimulationLabel'] = 'test_'
            data['file_config']['SimulationDir'] = '.'

            # Entramos a un directorio temporal
            with TemporaryDirectory() as tempdir:
                os.chdir(tempdir)
                # Llamamos a hector con la configuración en un diccionario,
                # que la función oportunamente escribe en un archivo, pero que
                # no hay problema porque se va a destruir junto con el
                # directorio temporal.

                # Esta función tiene side efects, asi que aunque devuelve el
                # diccionario no me gasto en tomarlo, para que nos acordemos
                # que tiene side effects y que eso puede ser malo.
                call_hector_with_toml(data,return_type='hash')

                # Agregamos la data a una lista.
                data_to_write.append(data['NoiseModels'])

    # Volver a donde estabamos
    os.chdir(cwdir)
    # Escribir el resultado.
    with open('test_cases.toml','w') as f:
        # Cada toml sale por separado, pero después se va a poder leer todo
        # junto porque cada modelo tiene una etiqueta.
        lines_to_write = map(toml.dumps,data_to_write)
        f.writelines(lines_to_write)

    return 0

if __name__ == "__main__":
    # La semilla del generador y el número de ejemplos a generar esta
    # hardcodeado aunque podría no estarlo.
    np.random.seed(0)
    exit(main(3))

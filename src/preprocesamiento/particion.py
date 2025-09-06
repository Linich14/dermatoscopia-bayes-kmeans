"""
Funciones para partición de datos por imagen en conjuntos de entrenamiento, validación y test.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from config.configuracion import SEED

def particionar_datos(imagenes):
    """
    Realiza la partición de datos por imagen en tres conjuntos:
        - Entrenamiento (60%)
        - Validación (20%)
        - Test (20%)
    Usa una semilla fija para garantizar reproducibilidad.
    Devuelve tres listas estructuradas con los diccionarios de imagen.
    """
    if not imagenes:
        return [], [], []
    indices = np.arange(len(imagenes))
    train_idx, temp_idx = train_test_split(indices, test_size=0.4, random_state=SEED)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=SEED)
    train = [imagenes[i] for i in train_idx]
    val = [imagenes[i] for i in val_idx]
    test = [imagenes[i] for i in test_idx]
    return train, val, test

"""
Clasificación simulada usando K-Means (base para implementar).
"""
import numpy as np

def clasificar_kmeans(imagen):
    """
    Clasifica los píxeles de una imagen usando K-Means (simulado).
    Retorna un array con la categoría asignada a cada píxel.
    """
    arr = np.array(imagen.convert('L'))
    # Pendiente de implementación